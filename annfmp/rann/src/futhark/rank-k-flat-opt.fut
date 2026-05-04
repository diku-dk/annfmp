let imap  as f = map f as
let imap2 as bs f = map2 f as bs
let imap3 as bs cs f = map3 f as bs cs
let imap4 as bs cs ds f = map4 f as bs cs ds
let ifilter as p = filter p as

let partition3 [n] (p: f32) (A: [n]f32) : (i64, i64, [n]f32) =
  let conds = imap A (\ x -> if x < p then (1i64,0i64,0i64) else if x == p then (0,1,0) else (0,0,1) )
  let (sc_1, sc_2, sc_3) = scan (\ (a1,a2,a3) (b1,b2,b3) -> (a1+b1, a2+b2, a3+b3) )
                                (0i64, 0i64, 0i64) conds |> unzip3
  let lt_len = last sc_1
  let eq_len = last sc_2
  let inds = 
    map4 (\ i1 i2 i3 x ->
            if x < p then i1 - 1
            else if x == p
                 then lt_len + i2 - 1
                 else lt_len + eq_len + i3 - 1
         ) sc_1 sc_2 sc_3 A 
  in  (lt_len, eq_len, scatter (replicate n 0f32) inds A)

let mkFlagArray 't [m] 
            (aoa_shp: [m]i32) (zero: t)
            (aoa_val: [m]t  ) : []t =
  let shp_scn = scan (+) 0 aoa_shp
  let aoa_len = shp_scn[m-1]
  let shp_ind = imap2 aoa_shp (indices aoa_shp)
                      (\ s i ->
                         if s==0 then -1i64
                         else if i==0 then 0i64
                         else i64.i32 shp_scn[i-1]
                      )
  let flags = scatter (replicate (i64.i32 aoa_len) zero)
                      shp_ind aoa_val
  in flags

let mkII1 [m] (shp: [m]i32) : *[]i32 =
    let flags = mkFlagArray shp 0i8 (replicate m 1i8)
    in  map i32.i8 flags
     |> scan (+) 0i32

let ones [q] 't (_xs: [q]t) = replicate q 1i32

-- meds: hopefully a decent estimate of the median values for each partition
-- ks:   the k-th smallest element to be searched for each partition (starting from 1)
-- shp, II1, A:  the rep of the iregular array: shape, II1-helper (plus 1) and flat data
-- More precisely, if we have m segments II1 will have the same length as the flat A,
--   and each element will indicate the segment (number plus one) in which the current
--   element of A rezides.
-- E.g., assuming shp = [3,5,7], then the length of A and II1 is 3+5+7=15,
--       if we want the median, ks = [2, 3, 4], and
--       II1 = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3].
--
let rankSearchBatch [m][n] (meds: [m]f32) (ks: [m]i32)
                           (shp: [m]i32) (II1: *[n]i32) (A: *[n]f32) : [m]f32 =
  let II1_bak = replicate n 0i32
  let A_bak = replicate n 0f32
  let res = replicate m 0f32
  let q = 0i64

  let (_, _shp, _,_,_,_,_, res) =
    loop (ks : [m]i32, shp : [m]i32, II1, II1_bak, A, A_bak, q, res)
      while (length A > 0) do
        -- compute helpers based on shape
        let shp_sc = scan (+) 0 shp

        -- F(let pivot = last A)
        let pivots =
            imap2 shp_sc (indices shp_sc)
              (\ off i -> if q == 0i64
                          then meds[i]
                          else if off == 0
                               then 0f32
                               else A[off - 1]
              ) |> opaque

        -- compute lt_len and eq_len by means of histograms:
        let h_inds =
            imap2 II1 A
                  (\ sgmindp1 a ->
                    let sgmind = sgmindp1 - 1
                    let pivot  = pivots[sgmind]
                    let h_ind  = sgmind << 1
                    in  i64.i32 <|
                          if a < pivot then h_ind
                          else if pivot == a then h_ind + 1
                          else -1i32
                  )
        let h_vals = ones A
        let lens = reduce_by_index (replicate (2*m) 0i32) (+) 0i32 h_inds h_vals

        --
        let (shp', kinds, ks') =
          imap2 ks (indices ks)
            (\ k i ->
                if k < 0 then (0, 3i8, -1) -- already processed
                else let lt_len = lens[i << 1] in
                     if k < lt_len then (lt_len, 0i8, k)
                     else let eq_len = lens[ (i << 1) + 1]
                          let lteq_len = lt_len + eq_len in
                          if k < lteq_len then (0, 1i8, -1)
                          else (shp[i] - lteq_len, 2i8, k - lteq_len)
            )
          |> unzip3

        -- write the subarrays that have finished
        let (scat_inds, scat_vals) =
            imap2 (indices kinds) kinds
                  (\ i knd ->
                    if knd == 1i8
                    then (i, pivots[i])
                    else (-1, 0.0)
                  )
            |> unzip
        let res' = scatter res scat_inds scat_vals

        -- use a filter to extract elements
        let keepElem sgmindp1 a =
                let sgmind = sgmindp1 - 1
                let pivot = pivots[sgmind]
                let kind  =  kinds[sgmind] in
                if (a < pivot && kind == 0) then true
                else if (a > pivot && kind == 2) then true
                else false

        let conds = map2 keepElem II1 A |> opaque -- strange fusion with duplicating computation

        let tmp_inds = map i32.bool conds
                    |> scan (+) 0i32
        let tot_len = i64.i32 (last tmp_inds)
        let scat_inds = imap2 conds tmp_inds
              (\ c ind -> if c then i64.i32 (ind-1) else -1i64)
        let A'   = scatter A_bak scat_inds A
        let II1' = scatter II1_bak scat_inds II1
        let II1''= II1'[:tot_len]
        let A''  = A'[:tot_len]
        
        in  (ks', shp', II1'', II1, A'', A, q+1, res')
  in res

-- ==
-- entry: main, test
-- random input { [131072][1024]f32 }
-- random input { [128][1048576]f32 }

entry main [m][n] (ass: [m][n]f32) =
--    let means = map (reduce_comm (+) 0f32) ass |> map (/ (f32.i64 n))
    let mins = map (reduce_comm f32.min f32.highest) ass
    let maxs = map (reduce_comm f32.max f32.lowest ) ass
    let means = map2 (+) mins maxs |> map (/ 2) |> opaque

    let k  = i32.i64 (n / 2)
    let ks = replicate m k
    let N  = m * n
    let shp = replicate m (i32.i64 n)

    -- II1 should in principle be computed with (mkII1 shp)
    let II1 = (tabulate_2d m n (\i _j -> i32.i64 i + 1) |> flatten) :> *[N]i32
    let A = copy (flatten ass) :> *[N]f32
    let medians = rankSearchBatch means ks shp II1 A
    in (mins, maxs, means, medians)

entry test [m][n] (ass: [m][n]f32) =
    --let mins = map (reduce_comm f32.min f32.highest) ass
    --let maxs = map (reduce_comm f32.max f32.lowest ) ass
    --let meds = map2 (+) mins maxs |> map (/ 2) |> opaque

    let means = map (reduce_comm (+) 0f32) ass
    let k  = i32.i64 (n / 2)
    let ks = replicate m k
    let N  = m * n
    let shp = replicate m (i32.i64 n)

    -- II1 should in principle be computed with (mkII1 shp)
    let II1 = (tabulate_2d m n (\i _j -> i32.i64 i + 1) |> flatten) :> *[N]i32
    let A = copy (flatten ass) :> *[N]f32
    let medians = rankSearchBatch means ks shp II1 A

    -- checking correctness
    -- test correctness
    in imap3 (map i64.i32 ks) medians ass
             (\ k med as -> 
                let (lt_len, eq_len, _) = partition3 med as
                -- in  (lt_len, eq_len, k)
                in  k >= lt_len && k < lt_len + eq_len
             )  --|> unzip3
       |> reduce (&&) true
