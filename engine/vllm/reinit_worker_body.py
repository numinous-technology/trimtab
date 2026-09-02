    def trimtab_release_kv(self) -> dict:
        """trimtab warm reinit, worker side. Free the KV pool and drop the
        captured CUDA graphs. Weights stay on the GPU.

        The KV cache is allocated inside CuMemAllocator.use_memory_pool(
        tag="kv_cache"). Dropping the tensor references marks the blocks free,
        but torch.cuda.empty_cache() errors on a pluggable allocator
        (pytorch/pytorch#145168), so the pages are not returned. vLLM's own
        use_memory_pool exit works around this by snapshotting the pool and
        releasing every allocated_size==0 block by hand. We run the same
        release here, which returns the physical pages cleanly (single unmap,
        popped from pointer_to_data) with no double-unmap. discard() is the
        wrong primitive: it marks blocks asleep and later double-unmaps.
        """
        import gc

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.device_allocator.cumem import CuMemAllocator

        mr = self.model_runner
        torch.cuda.synchronize()
        free0 = torch.cuda.mem_get_info()[0]

        try:
            from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

            BreakableCUDAGraphWrapper.clear_all_graphs()
        except Exception:
            pass
        CUDAGraphWrapper.clear_all_graphs()
        for key_set in mr.cudagraph_dispatcher.cudagraph_keys.values():
            key_set.clear()
        mr.cudagraph_dispatcher.keys_initialized = False

        # Drop every KV reference (attention KV and hybrid mamba state both live
        # in mr.kv_caches and layer.kv_cache, which this clears).
        mr._cleanup_profiling_kv_cache()
        gc.collect()

        # Tear down the kv_cache MemPool itself, two-phase, the way vLLM's own
        # release_pools does. The MemPool destructor releases its blocks through
        # the pluggable free callback (single clean unmap each). Dropping the
        # MemPool while the allocator is still strongly held avoids the
        # finalize-order crash (pytorch/pytorch#145168). Manual per-block frees
        # instead race the destructor and double-free (MMU fault).
        allocator = CuMemAllocator.get_instance()
        data = allocator.allocator_and_pools.pop("kv_cache", None)
        released = 0
        if data is not None:
            mem_pool, pool_alloc = data
            for allocation in mem_pool.snapshot():
                if allocation["allocated_size"] == 0:
                    released += allocation.get("total_size", 0)
            del data
            del mem_pool          # phase 1: ~MemPool runs, allocator still alive
            gc.collect()
            del pool_alloc        # phase 2: drop the pluggable allocator
            gc.collect()
        torch.cuda.synchronize()

        # vLLM caps the process at gpu_memory_utilization through
        # set_per_process_memory_fraction. Lift it, the profiler sizes the new
        # pool against device free memory, not this guard.
        torch.cuda.set_per_process_memory_fraction(1.0)

        # init_snapshot stays as captured at boot (before weights loaded). The
        # profiler measures non_kv_cache_memory as consumption relative to it,
        # which must include the weights. Refreshing it post-weights would drop
        # the weights from that accounting and oversize the new pool.
        return {"released_gib": round(released / 2**30, 2),
                "free_gib": round((torch.cuda.mem_get_info()[0] - free0) / 2**30, 2)}

