/*
    dumpster, a cycle-tracking garbage collector for Rust.    Copyright (C) 2023 Clayton Ramsey.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

//! A synchronized collection algorithm.

use std::{
    alloc::{dealloc, Layout},
    cell::{Cell, LazyCell, RefCell},
    collections::hash_map::Entry,
    hash::Hash,
    mem::{replace, take, transmute},
    ops::Range,
    ptr::{drop_in_place, NonNull},
};

#[cfg(not(loom))]
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use foldhash::{HashMap, HashMapExt};

#[cfg(loom)]
use loom::{
    lazy_static,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
    thread_local,
};

#[cfg(not(loom))]
use parking_lot::{Mutex, RwLock};

#[cfg(loom)]
use crate::sync::loom_ext::{Mutex, RwLock};

use crate::{ptr::Erased, Trace, Visitor};

use super::{default_collect_condition, CollectCondition, CollectInfo, Gc, GcBox, CURRENT_TAG};

/// The garbage truck, which is a global data structure containing information about allocations
/// which might need to be collected.
struct GarbageTruck {
    /// The contents of the garbage truck, containing all the allocations which need to be
    /// collected and have already been delivered by a [`Dumpster`].
    contents: Mutex<LazyCell<HashMap<AllocationId, TrashCan>>>,
    /// A lock used for synchronizing threads that are awaiting completion of a collection process.
    /// This lock should be acquired for reads by threads running a collection and for writes by
    /// threads awaiting collection completion.
    collecting_lock: RwLock<()>,
    /// The number of [`Gc`]s dropped since the last time [`GarbageTruck::collect_all()`] was
    /// called.
    n_gcs_dropped: AtomicUsize,
    /// The number of [`Gc`]s currently existing (which have not had their internals replaced with
    /// `None`).
    n_gcs_existing: AtomicUsize,
    /// The function which determines whether a collection should be triggered.
    /// This pointer value should always be cast to a [`CollectCondition`], but since `AtomicPtr`
    /// doesn't handle function pointers correctly, we just cast to `*mut ()`.
    collect_condition: AtomicPtr<()>,
}

/// A structure containing the global information for the garbage collector.
pub(super) struct Dumpster {
    /// A lookup table for the allocations which may need to be cleaned up later.
    pub contents: RefCell<HashMap<AllocationId, TrashCan>>,
    /// The number of times an allocation on this thread has been dropped.
    n_drops: Cell<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
/// A unique identifier for an allocation.
pub(super) struct AllocationId(NonNull<GcBox<()>>);

#[derive(Debug)]
/// The information which describes an allocation that may need to be cleaned up later.
pub(super) struct TrashCan {
    /// A pointer to the allocation to be cleaned up.
    ptr: Erased,
    /// The function which inserts the allocation into the reference graph and enqueues it for
    /// child snapshotting.
    /// This function is safe to call on `ptr`.
    init_fn: unsafe fn(
        Erased,
        &mut HashMap<AllocationId, AllocationInfo>,
        &mut Vec<AllocationId>,
        &mut Vec<WorkItem>,
    ),
}

#[derive(Debug)]
/// A node in the reference graph, which is constructed while searching for unreachable allocations.
struct AllocationInfo {
    /// An erased pointer to the allocation.
    ptr: Erased,
    /// Function for dropping the allocation when its weak and strong count hits zero.
    /// Should have the same behavior as dropping a Gc normally to a reference count of zero.
    weak_drop_fn: unsafe fn(Erased),
    /// Information about this allocation's reachability.
    reachability: Reachability,
}

#[derive(Debug)]
/// The state of whether an allocation is reachable or of unknown reachability.
enum Reachability {
    /// The information describing an allocation whose accessibility is unknown.
    Unknown {
        /// The range within the children slab for the allocations directly accessible from this
        /// allocation.
        children_range: Range<usize>,
        /// The number of references in the reference count for this allocation which are
        /// "unaccounted," which have not been found while constructing the graph.
        /// It is the difference between the allocations indegree in the "true" reference graph vs
        /// the one we are currently building.
        n_unaccounted: usize,
        /// A function used to destroy the allocation.
        destroy_fn: unsafe fn(Erased, &HashMap<AllocationId, AllocationInfo>),
    },
    /// The allocation here is reachable.
    /// No further information is needed.
    Reachable,
}

/// A pending work item for the iterative DFS work queue.
///
/// Each item represents an allocation that has been inserted into the reference graph but whose
/// children have not yet been snapshotted into the children slab.
#[derive(Debug)]
struct WorkItem {
    /// The allocation to snapshot.
    id: AllocationId,
    /// The type-erased snapshot function, monomorphized over the allocation's concrete type.
    snapshot_fn: unsafe fn(
        AllocationId,
        &mut HashMap<AllocationId, AllocationInfo>,
        &mut Vec<AllocationId>,
        &mut Vec<WorkItem>,
    ),
}

#[cfg(not(loom))]
/// The global garbage truck.
/// All [`TrashCan`]s should eventually end up in here.
static GARBAGE_TRUCK: GarbageTruck = GarbageTruck::new();

#[cfg(loom)]
lazy_static! {
    static ref GARBAGE_TRUCK: GarbageTruck = GarbageTruck::new();
}

thread_local! {
    /// The dumpster for this thread.
    /// Allocations which are "dirty" will be transferred to this dumpster before being moved into
    /// the garbage truck for final collection.
    pub(super) static DUMPSTER: Dumpster = Dumpster {
        contents: RefCell::new(HashMap::new()),
        n_drops: Cell::new(0),
    };
}

#[cfg(not(loom))]
thread_local! {
    /// Whether the currently-running thread is doing a cleanup.
    /// This cannot be stored in `DUMPSTER` because otherwise it would cause weird use-after-drop
    /// behavior.
    static CLEANING: Cell<bool> = const { Cell::new(false) };
}

#[cfg(loom)]
thread_local! {
    /// Whether the currently-running thread is doing a cleanup.
    /// This cannot be stored in `DUMPSTER` because otherwise it would cause weird use-after-drop
    /// behavior.
    static CLEANING: Cell<bool> = Cell::new(false);
}

thread_local! {
    /// Reusable flat slab of child [`AllocationId`]s, retained between collections to amortize
    /// allocation cost.  Each node in the reference graph stores a [`Range<usize>`] into this
    /// slab rather than its own heap-allocated [`Vec`].
    static CHILDREN_SLAB: RefCell<Vec<AllocationId>> = const { RefCell::new(Vec::new()) };

    /// Reusable reference graph, retained between collections to amortize the cost of
    /// re-allocating the [`HashMap`]'s backing array each sweep.
    /// Must be cleared (not just taken) before use, since entries from a prior sweep have
    /// dangling pointers after their GcBoxes are freed.
    static REF_GRAPH: RefCell<HashMap<AllocationId, AllocationInfo>> =
        RefCell::new(HashMap::new());

    /// Reusable work queue for the iterative DFS, retained between collections.
    /// Always empty when returned to storage (drained by the work loop in `collect_all`).
    static WORK_QUEUE: RefCell<Vec<WorkItem>> = const { RefCell::new(Vec::new()) };
}

/// Collect all allocations in the garbage truck (but not necessarily the dumpster), then await
/// completion of the collection.
/// Ensures that all allocations dropped on the calling thread are cleaned up
pub fn collect_all_await() {
    _ = DUMPSTER.try_with(|d| d.deliver_to(&GARBAGE_TRUCK));
    GARBAGE_TRUCK.collect_all();
    drop(GARBAGE_TRUCK.collecting_lock.read());
}

/// Notify that a `Gc` was destroyed, and update the tracking count for the number of dropped and
/// existing `Gc`s.
///
/// This may trigger a linear-time cleanup of all allocations, but this will be guaranteed to
/// occur with less-than-linear frequency, so it's always O(1).
pub fn notify_dropped_gc() {
    GARBAGE_TRUCK.n_gcs_existing.fetch_sub(1, Ordering::Relaxed);
    GARBAGE_TRUCK.n_gcs_dropped.fetch_add(1, Ordering::Relaxed);

    // Do not do deliver or collect if we are currently cleaning or this thread is dying.
    // This prevents deadlocks.
    if !CLEANING.try_with(Cell::get).is_ok_and(|x| !x) {
        return;
    }

    _ = DUMPSTER.try_with(|dumpster| {
        dumpster.n_drops.set(dumpster.n_drops.get() + 1);
        if dumpster.is_full() {
            dumpster.deliver_to(&GARBAGE_TRUCK);
        }
    });

    let collect_cond = unsafe {
        // SAFETY: we only ever store collection conditions in the collect-condition box
        transmute::<*mut (), CollectCondition>(
            GARBAGE_TRUCK.collect_condition.load(Ordering::Relaxed),
        )
    };
    if collect_cond(&CollectInfo { _private: () }) {
        GARBAGE_TRUCK.collect_all();
    }
}

/// Notify that a [`Gc`] was created, and increment the number of total existing `Gc`s.
pub fn notify_created_gc() {
    GARBAGE_TRUCK.n_gcs_existing.fetch_add(1, Ordering::Relaxed);
}

/// Mark an allocation as "dirty," implying that it may or may not be inaccessible and need to
/// be cleaned up.
///
/// # Safety
///
/// When calling this method, you have to ensure that `allocation`
/// is [convertible to a reference](core::ptr#pointer-to-reference-conversion).
pub(super) unsafe fn mark_dirty<T>(allocation: NonNull<GcBox<T>>)
where
    T: Trace + Send + Sync + ?Sized,
{
    _ = DUMPSTER.try_with(|dumpster| {
        if dumpster
            .contents
            .borrow_mut()
            .insert(
                AllocationId::from(allocation),
                TrashCan {
                    ptr: Erased::new(allocation),
                    init_fn: init_root::<T>,
                },
            )
            .is_none()
        {
            // SAFETY: the caller must guarantee that `allocation` meets all the
            // requirements for a reference.
            unsafe { allocation.as_ref() }
                .weak
                .fetch_add(1, Ordering::Acquire);
        }
    });
}

/// Mark an allocation as "clean," implying that it has already been cleaned up and does not
/// need to be cleaned again.
pub(super) fn mark_clean<T>(allocation: &GcBox<T>)
where
    T: Trace + Send + Sync + ?Sized,
{
    _ = DUMPSTER.try_with(|dumpster| {
        if dumpster
            .contents
            .borrow_mut()
            .remove(&AllocationId::from(allocation))
            .is_some()
        {
            allocation.weak.fetch_sub(1, Ordering::Release);
        }
    });
}

#[cfg(test)]
/// Deliver all [`TrashCan`]s from this thread's dumpster into the garbage truck.
///
/// This function is available to to support testing, but currently is not part of the public API.
pub(super) fn deliver_dumpster() {
    _ = DUMPSTER.try_with(|d| d.deliver_to(&GARBAGE_TRUCK));
}

/// Set the function which determines whether the garbage collector should be run.
///
/// `f` will be periodically called by the garbage collector to determine whether it should perform
/// a full traversal of the heap.
/// When `f` returns true, a traversal will begin.
///
/// # Examples
///
/// ```
/// use dumpster::sync::{set_collect_condition, CollectInfo};
///
/// /// This function will make sure a GC traversal never happens unless directly activated.
/// fn never_collect(_: &CollectInfo) -> bool {
///     false
/// }
///
/// set_collect_condition(never_collect);
/// ```
pub fn set_collect_condition(f: CollectCondition) {
    GARBAGE_TRUCK
        .collect_condition
        .store(f as *mut (), Ordering::Relaxed);
}

/// Get the number of `[Gc]`s dropped since the last collection.
pub fn n_gcs_dropped() -> usize {
    GARBAGE_TRUCK.n_gcs_dropped.load(Ordering::Relaxed)
}

/// Get the number of `[Gc]`s currently existing in the entire program.
pub fn n_gcs_existing() -> usize {
    GARBAGE_TRUCK.n_gcs_existing.load(Ordering::Relaxed)
}

impl Dumpster {
    /// Deliver all [`TrashCan`]s contained by this dumpster to the garbage collect, removing them
    /// from the local dumpster storage and adding them to the global truck.
    fn deliver_to(&self, garbage_truck: &GarbageTruck) {
        let mut guard = garbage_truck.contents.lock();
        self.n_drops.set(0);
        self.deliver_to_contents(&mut guard);
    }

    /// Deliver the entries in this dumpster to `contents`.
    fn deliver_to_contents(&self, contents: &mut HashMap<AllocationId, TrashCan>) {
        for (id, can) in self.contents.borrow_mut().drain() {
            if contents.insert(id, can).is_some() {
                unsafe {
                    // SAFETY: an allocation can only be in the dumpster if it still exists and its
                    // header is valid
                    id.0.as_ref()
                }
                .weak
                .fetch_sub(1, Ordering::Release);
            }
        }
    }

    /// Determine whether this dumpster is full (and therefore should have its contents delivered to
    /// the garbage truck).
    fn is_full(&self) -> bool {
        self.contents.borrow().len() > 100_000 || self.n_drops.get() > 100_000
    }
}

impl GarbageTruck {
    /// Construct a new, empty garbage truck.
    ///
    /// Since the `GarbageTruck` is meant to be a single global value, this function should only be
    /// called once in the initialization of `GARBAGE_TRUCK`.
    #[cfg(not(loom))]
    const fn new() -> Self {
        Self {
            contents: Mutex::new(LazyCell::new(HashMap::new)),
            collecting_lock: RwLock::new(()),
            n_gcs_dropped: AtomicUsize::new(0),
            n_gcs_existing: AtomicUsize::new(0),
            collect_condition: AtomicPtr::new(default_collect_condition as *mut ()),
        }
    }

    /// Construct a new, empty garbage truck.
    ///
    /// Since the `GarbageTruck` is meant to be a single global value, this function should only be
    /// called once in the initialization of `GARBAGE_TRUCK`.
    #[cfg(loom)]
    fn new() -> Self {
        Self {
            contents: Mutex::new(LazyCell::new(HashMap::new)),
            collecting_lock: RwLock::new(()),
            n_gcs_dropped: AtomicUsize::new(0),
            n_gcs_existing: AtomicUsize::new(0),
            collect_condition: AtomicPtr::new(default_collect_condition as *mut ()),
        }
    }

    /// Search through the set of existing allocations which have been marked inaccessible, and see
    /// if they are inaccessible.
    /// If so, drop those allocations.
    fn collect_all(&self) {
        let collecting_guard = self.collecting_lock.write();
        self.n_gcs_dropped.store(0, Ordering::Relaxed);

        let to_collect = take(&mut **self.contents.lock());

        // Take reusable buffers from thread-local storage, retaining their heap allocations
        // across collections to avoid re-allocating them every sweep.
        let mut ref_graph = REF_GRAPH.with(|r| take(&mut *r.borrow_mut()));
        ref_graph.clear(); // must clear: prior entries contain dangling pointers after GC frees
        ref_graph.reserve(to_collect.len());

        let mut slab = CHILDREN_SLAB.with(|s| take(&mut *s.borrow_mut()));
        slab.clear();

        let mut work_queue = WORK_QUEUE.with(|w| take(&mut *w.borrow_mut()));
        // work_queue is always empty when returned (drained by the loop below)

        CURRENT_TAG.fetch_add(1, Ordering::Release);

        for (_, TrashCan { ptr, init_fn }) in to_collect {
            unsafe {
                // SAFETY: `ptr` may only be in `to_collect` if it was a valid pointer
                // and `init_fn` must have been created with the intent of referring to
                // the erased type of `ptr`.
                init_fn(ptr, &mut ref_graph, &mut slab, &mut work_queue);
            }
        }

        // Drain the work queue, snapshotting each allocation's children into the slab.
        while let Some(WorkItem { id, snapshot_fn }) = work_queue.pop() {
            unsafe {
                // SAFETY: `id` was inserted into `ref_graph` by `init_root` or `Dfs::visit_sync`,
                // and `snapshot_fn` is monomorphized for its concrete type.
                snapshot_fn(id, &mut ref_graph, &mut slab, &mut work_queue);
            }
        }

        let root_ids = ref_graph
            .iter()
            .filter_map(|(&k, v)| match v.reachability {
                Reachability::Reachable => Some(k),
                Reachability::Unknown { n_unaccounted, .. } => (n_unaccounted > 0
                    || unsafe {
                        // SAFETY: we found `k` in the reference graph,
                        // so it must still be an extant allocation
                        k.0.as_ref().weak.load(Ordering::Acquire) > 1
                    })
                .then_some(k),
            })
            .collect::<Vec<_>>();
        for root_id in root_ids {
            mark(root_id, &mut ref_graph, &slab);
        }

        CLEANING.with(|c| c.set(true));
        // set of allocations which must be destroyed because we were the last weak pointer to it
        let mut weak_destroys = Vec::new();
        for (id, node) in &ref_graph {
            let header_ref = unsafe { id.0.as_ref() };
            match node.reachability {
                Reachability::Unknown { destroy_fn, .. } => unsafe {
                    // SAFETY: `destroy_fn` must have been created with `node.ptr` in mind,
                    // and we have proven that no other references to `node.ptr` exist
                    destroy_fn(node.ptr, &ref_graph);
                },
                Reachability::Reachable => {
                    if header_ref.weak.fetch_sub(1, Ordering::Release) == 1
                        && header_ref.strong.load(Ordering::Acquire) == 0
                    {
                        // we are the last reference to the allocation.
                        // mark to be cleaned up later
                        // no real synchronization loss to storing the guard because we had the last
                        // reference anyway
                        weak_destroys.push((node.weak_drop_fn, node.ptr));
                    }
                }
            }
        }
        CLEANING.with(|c| c.set(false));
        for (drop_fn, ptr) in weak_destroys {
            unsafe {
                // SAFETY: we have proven (via header_ref.weak = 1) that the cleaning
                // process had the last reference to the allocation.
                // `drop_fn` must have been created with the true value of `ptr` in mind.
                drop_fn(ptr);
            };
        }

        // Return buffers to thread-local storage, retaining their capacity for the next collection.
        ref_graph.clear(); // clear again: entries now have dangling pointers after destruction
        REF_GRAPH.with(|r| *r.borrow_mut() = ref_graph);
        CHILDREN_SLAB.with(|s| *s.borrow_mut() = slab);
        WORK_QUEUE.with(|w| *w.borrow_mut() = work_queue);

        drop(collecting_guard);
    }
}

/// Insert a root allocation into the reference graph and enqueue it for child snapshotting.
///
/// This is called once per entry in `to_collect` at the start of each collection sweep.
///
/// # Safety
///
/// `ptr` must have been created as a pointer to a `GcBox<T>`.
unsafe fn init_root<T: Trace + Send + Sync + ?Sized>(
    ptr: Erased,
    ref_graph: &mut HashMap<AllocationId, AllocationInfo>,
    _slab: &mut Vec<AllocationId>,
    work_queue: &mut Vec<WorkItem>,
) {
    let box_ref = unsafe {
        // SAFETY: We require `ptr` to be an erased pointer to `GcBox<T>`.
        ptr.specify::<GcBox<T>>().as_ref()
    };
    let starting_id = AllocationId::from(box_ref);
    let Entry::Vacant(v) = ref_graph.entry(starting_id) else {
        // the weak count was incremented by another DFS operation elsewhere.
        // Decrement it to have only one from us.
        box_ref.weak.fetch_sub(1, Ordering::Release);
        return;
    };
    let strong_count = box_ref.strong.load(Ordering::Acquire);
    v.insert(AllocationInfo {
        ptr,
        weak_drop_fn: drop_weak_zero::<T>,
        reachability: Reachability::Unknown {
            children_range: 0..0, // filled in by snapshot_node
            n_unaccounted: strong_count,
            destroy_fn: destroy_erased::<T>,
        },
    });
    work_queue.push(WorkItem {
        id: starting_id,
        snapshot_fn: snapshot_node::<T>,
    });
}

/// Snapshot all direct children of `id` into the slab, updating the reference graph.
///
/// All children are appended contiguously to `slab` and the node's `children_range` is updated.
/// Newly discovered children are inserted into `ref_graph` and enqueued in `work_queue`.
///
/// # Safety
///
/// `id` must refer to a `GcBox<T>` that is present in `ref_graph`.
unsafe fn snapshot_node<T: Trace + Send + Sync + ?Sized>(
    id: AllocationId,
    ref_graph: &mut HashMap<AllocationId, AllocationInfo>,
    slab: &mut Vec<AllocationId>,
    work_queue: &mut Vec<WorkItem>,
) {
    // Copy the raw pointer out of the map so we don't hold a borrow while mutating ref_graph.
    let ptr = ref_graph[&id].ptr;
    let box_ref = unsafe {
        // SAFETY: caller guarantees `id` refers to a valid `GcBox<T>`.
        ptr.specify::<GcBox<T>>().as_ref()
    };

    // Load current_tag once here; the Dfs visitor caches it to avoid per-edge atomic loads.
    let current_tag = CURRENT_TAG.load(Ordering::Relaxed);

    let children_start = slab.len();
    // Scope the visitor so that its mutable borrows of `slab`, `ref_graph`, and `work_queue`
    // are released before we read `slab.len()` and mutate `ref_graph` below.
    let (accept_result, mark_current) = {
        let mut visitor = Dfs {
            ref_graph,
            slab,
            current_id: id,
            mark_current: false,
            work_queue,
            current_tag,
        };
        let result = box_ref.value.accept(&mut visitor);
        (result, visitor.mark_current)
    };
    let children_end = slab.len();

    let needs_mark = accept_result.is_err()
        || box_ref.generation.load(Ordering::Acquire) >= current_tag
        || mark_current;

    // Record the contiguous range of children for this node.
    if let Some(AllocationInfo {
        reachability:
            Reachability::Unknown {
                ref mut children_range,
                ..
            },
        ..
    }) = ref_graph.get_mut(&id)
    {
        *children_range = children_start..children_end;
    }

    if needs_mark {
        mark(id, ref_graph, slab);
    }
}

#[derive(Debug)]
/// The visitor structure used for building the found-reference-graph of allocations.
///
/// Unlike the old recursive visitor, this one only snapshots the immediate children of the current
/// node into the slab without recursing.  Recursion is replaced by the iterative work queue in
/// [`GarbageTruck::collect_all`].
pub(super) struct Dfs<'a> {
    /// The reference graph.
    /// Each allocation is assigned a node.
    ref_graph: &'a mut HashMap<AllocationId, AllocationInfo>,
    /// Flat slab of child allocation IDs shared across all nodes in this collection sweep.
    slab: &'a mut Vec<AllocationId>,
    /// The allocation ID currently being visited.
    current_id: AllocationId,
    /// Set to `true` if a tag/generation check on any child indicates the current node is
    /// reachable from outside the candidate set.
    mark_current: bool,
    /// Work queue: allocations whose children still need to be snapshotted.
    work_queue: &'a mut Vec<WorkItem>,
    /// The value of [`CURRENT_TAG`] at the start of the current collection sweep, cached here
    /// to avoid one atomic load per edge traversal in [`Visitor::visit_sync`].
    current_tag: usize,
}

impl Visitor for Dfs<'_> {
    fn visit_sync<T>(&mut self, gc: &Gc<T>)
    where
        T: Trace + Send + Sync + ?Sized,
    {
        if Gc::is_dead(gc) {
            return;
        }
        // must not use deref operators since we don't want to update the generation
        let ptr = unsafe {
            // SAFETY: This is the same as the deref implementation, but avoids
            // incrementing the generation count.
            gc.ptr.get().unwrap()
        };
        let box_ref = unsafe {
            // SAFETY: same as above.
            ptr.as_ref()
        };
        if gc.tag.swap(self.current_tag, Ordering::Relaxed) >= self.current_tag
            || box_ref.generation.load(Ordering::Acquire) >= self.current_tag
        {
            // This pointer was already tagged by this sweep, so it must have been moved by
            // a concurrent thread — the current node is reachable.
            self.mark_current = true;
            return;
        }

        // Optimization: if the current node has already been proven reachable (e.g. by a
        // concurrent marking propagated from an ancestor), there is no point building its
        // children list further.
        if !matches!(
            self.ref_graph[&self.current_id].reachability,
            Reachability::Unknown { .. }
        ) {
            return;
        }

        let new_id = AllocationId::from(box_ref);

        // Record this child in the slab (the range is finalized by snapshot_node after accept).
        self.slab.push(new_id);

        match self.ref_graph.entry(new_id) {
            Entry::Occupied(mut o) => match o.get_mut().reachability {
                Reachability::Unknown {
                    ref mut n_unaccounted,
                    ..
                } => {
                    *n_unaccounted -= 1;
                }
                Reachability::Reachable => (),
            },
            Entry::Vacant(v) => {
                // This allocation has never been visited by the reference graph builder.
                let strong_count = box_ref.strong.load(Ordering::Acquire);
                box_ref.weak.fetch_add(1, Ordering::Acquire);
                v.insert(AllocationInfo {
                    ptr: Erased::new(ptr),
                    weak_drop_fn: drop_weak_zero::<T>,
                    reachability: Reachability::Unknown {
                        children_range: 0..0, // filled in when this node is dequeued
                        n_unaccounted: strong_count - 1,
                        destroy_fn: destroy_erased::<T>,
                    },
                });
                // Enqueue for snapshotting rather than recursing immediately.
                self.work_queue.push(WorkItem {
                    id: new_id,
                    snapshot_fn: snapshot_node::<T>,
                });
            }
        }
    }

    fn visit_unsync<T>(&mut self, _: &crate::unsync::Gc<T>)
    where
        T: Trace + ?Sized,
    {
        unreachable!("sync Gc cannot own an unsync Gc");
    }
}

/// Traverse the reference graph, marking `root` and any allocations reachable from `root` as
/// reachable.
fn mark(
    root: AllocationId,
    graph: &mut HashMap<AllocationId, AllocationInfo>,
    slab: &[AllocationId],
) {
    let node = graph.get_mut(&root).unwrap();
    if let Reachability::Unknown { children_range, .. } =
        replace(&mut node.reachability, Reachability::Reachable)
    {
        for &child in &slab[children_range] {
            mark(child, graph, slab);
        }
    }
}

/// A visitor for decrementing the reference count of pointees.
pub(super) struct PrepareForDestruction<'a> {
    /// The reference graph.
    /// Must have been populated with reachability already.
    graph: &'a HashMap<AllocationId, AllocationInfo>,
}

impl Visitor for PrepareForDestruction<'_> {
    fn visit_sync<T>(&mut self, gc: &crate::sync::Gc<T>)
    where
        T: Trace + Send + Sync + ?Sized,
    {
        if Gc::is_dead(gc) {
            return;
        }
        let id = AllocationId::from(unsafe {
            // SAFETY: This is the same as dereferencing the GC.
            gc.ptr.get().unwrap()
        });
        if matches!(self.graph[&id].reachability, Reachability::Reachable) {
            unsafe {
                // SAFETY: This is the same as dereferencing the GC.
                id.0.as_ref().strong.fetch_sub(1, Ordering::Release);
            }
        }
        unsafe {
            // SAFETY: we have a unique reference to `gc` as we are destroying the structure.
            gc.kill();
        }
    }

    fn visit_unsync<T>(&mut self, _: &crate::unsync::Gc<T>)
    where
        T: Trace + ?Sized,
    {
        unreachable!("no unsync members of sync Gc possible!");
    }
}

/// Destroy an allocation, obliterating its GCs, dropping it, and deallocating it.
///
/// # Safety
///
/// `ptr` must have been created from a pointer to a `GcBox<T>`.
unsafe fn destroy_erased<T: Trace + Send + Sync + ?Sized>(
    ptr: Erased,
    graph: &HashMap<AllocationId, AllocationInfo>,
) {
    let specified = ptr.specify::<GcBox<T>>().as_mut();
    specified
        .value
        .accept(&mut PrepareForDestruction { graph })
        .expect("allocation assumed to be unreachable but somehow was accessed");
    let layout = Layout::for_value(specified);
    drop_in_place(specified);
    dealloc(std::ptr::from_mut::<GcBox<T>>(specified).cast(), layout);
}

/// Function for handling dropping an allocation when its weak and strong reference count reach
/// zero.
///
/// # Safety
///
/// `ptr` must have been created as a pointer to a `GcBox<T>`.
unsafe fn drop_weak_zero<T: Trace + Send + Sync + ?Sized>(ptr: Erased) {
    let mut specified = ptr.specify::<GcBox<T>>();
    assert_eq!(specified.as_ref().weak.load(Ordering::Relaxed), 0);
    assert_eq!(specified.as_ref().strong.load(Ordering::Relaxed), 0);

    let layout = Layout::for_value(specified.as_ref());
    drop_in_place(specified.as_mut());
    dealloc(specified.as_ptr().cast(), layout);
}

unsafe impl Send for AllocationId {}
unsafe impl Sync for AllocationId {}

impl<T> From<&GcBox<T>> for AllocationId
where
    T: Trace + Send + Sync + ?Sized,
{
    fn from(value: &GcBox<T>) -> Self {
        AllocationId(NonNull::from(value).cast())
    }
}

impl<T> From<NonNull<GcBox<T>>> for AllocationId
where
    T: Trace + Send + Sync + ?Sized,
{
    fn from(value: NonNull<GcBox<T>>) -> Self {
        AllocationId(value.cast())
    }
}

#[cfg(not(loom))] // cannot access lazy static in drop
impl Drop for Dumpster {
    fn drop(&mut self) {
        self.deliver_to(&GARBAGE_TRUCK);
        // collect_all();
    }
}
