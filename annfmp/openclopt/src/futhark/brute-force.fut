let sumSqrsSeq [d] (xs: [d]f32) (ys: [d]f32) : f32 =
    loop (res) = (0.0f32) for (x,y) in (zip xs ys) do
        let z = x-y in res + z*z

let bruteForce [m][d][k] (query: [d]f32) 
                         (knns0: [k](i32,f32))
                         (beg: i32, refs : [m][d]f32)
                       : [k](i32,f32) =
    loop (knns) = (copy knns0)
      for i < m do
        let dist = sumSqrsSeq query (refs[i]) in
        if dist > knns[k-1].1 then knns -- early exit
        else let ref_ind = i+beg in
             let (_, _, knns') =
               loop (dist, ref_ind, knns) for j < k do
                 let cur_nn = knns[j].1  in
                 if dist >= cur_nn
                 then (dist, ref_ind, knns)
                 else let tmp_ind = knns[j].0
                      let knns[j] = (ref_ind, dist)
                      let ref_ind = tmp_ind
                      in  (cur_nn, ref_ind, knns)
             in  knns'

let sortPartSortedSeqs [k] (knn: [k](i32,f32)) : [k](i32,f32) =
  -- now knn contains the neighbors in two partially ordered sequences:
  -- one starting at beginning and one starting at the end
  -- we need to sort them
  -- let knn = intrinsics.opaque (copy knn0)
  let (res, _, _) =
    loop (knn_sort, beg, end) = (replicate k (-1i32, f32.highest), 0, k-1)
      for i < k do
        let (next_el, beg', end') =
          if knn[beg].1 < knn[end].1
          then (knn[beg], beg+1, end)
          else (knn[end], beg, end-1)
        let knn_sort[i] = next_el
        in  (knn_sort, beg', end')
  in  res

let bruteForcePar [m][d][k] (query: [d]f32) 
                         (knn0: [k](i32,f32))
                         (beg: i32, refs : [m][d]f32)
                       : [k](i32,f32) =
  let knn = copy knn0
  let dists = map (sumSqrsSeq query) refs
  let cycle = true
  let j = 0i32
  let (_, knn, _, _) =
    loop (dists, knn, j, cycle)
      while cycle && (j < k) do
        let (min_ind, min_val) =
          reduce_comm (\ (i1,v1) (i2,v2) -> 
                        if v1 < v2 then (i1, v1) else
                        if v1 > v2 then (i2, v2) else
                        (if i1 <= i2 then i1 else i2, v1)
                      ) (m, f32.inf) (zip (iota m) dists)
        
        in  if min_val < (knn[k-1-j].1)
            then  let dists[min_ind] = f32.highest
                  let knn[k-1-j] = (beg+min_ind, min_val)
                  in  (dists, knn, j+1, true)
            else  (dists, knn, j, false)
  let knn_sort = sortPartSortedSeqs knn
  in  knn_sort

