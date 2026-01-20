# jaxpme.batched

This is an **experimental** subpackage to deal with *batched* Ewald summation. By this, we mean that we provide tools to perform Ewald summation on a number of different systems in parallel, by batching them all together and then doing scatter/gather operations. This is one way to accomplish this job: Another would be to pad to the biggest k-grid, stack the results, and then parallelise along the leading dimension. This is however inefficient if the systems have different k-grids, and it is hard to treat the case of non-periodic structures.
