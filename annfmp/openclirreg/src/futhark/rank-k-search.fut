import "util"

def log2 x = (loop (y,c) = (x,0i32) while y > 1i32 do (y >> 1, c+1)).1

def imap2 as bs f = map2 f as bs

def ones [q] 't (_xs: [q]t) = replicate q 1i32

def sgmscan 't [n] (op: t->t->t) (ne: t)
                   (flg : [n]i32) (arr : [n]t) : [n]t =
  let flgs_vals =
      scan ( \ (f1, x1) (f2,x2) ->
              let f = f1 | f2 in
              if f2 != 0 then (f, x2)
              else (f, op x1 x2) )
            (0,ne) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

-- | Segmented scan. Given a binary associative operator ``op`` with
-- neutral element ``ne``, computes the inclusive prefix scan of the
-- segments of ``as`` specified by the ``flags`` array, where `true`
-- starts a segment and `false` continues a segment.
def segmented_scan [n] 't
                   (op: t -> t -> t)
                   (ne: t)
                   (flags: [n]bool)
                   (as: [n]t) : *[n]t =
  (unzip (scan (\(x_flag, x) (y_flag, y) ->
                  ( x_flag || y_flag
                  , if y_flag then y else x `op` y
                  ))
               (false, ne)
               (zip flags as))).1

def seg_red [n] 't
                     (op: t -> t -> t)
                     (ne: t)
                     (flags: [n]bool)
                     (as: [n]t) 
                     (shp: []i64) =
  let indsp1 = scan (+) 0 shp
  let tmp = segmented_scan (op) ne flags as
  in map2(\s ip1 -> if s <= 0 then ne else tmp[ip1-1]) shp indsp1


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
def rankSearchBatch [m][n] (meds: [m]f32) (ks: [m]i32)
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

def computeMedianWithRankK [m] (shp: []i64) (input: [m]f32) (offsets: []i64) (flagArray: []i64) (II1: []i64) = 

  let shp = map i32.i64 shp

  -- Calculating mins
  let scanned_mins = sgmscan f32.min f32.highest (map i32.i64 flagArray) input
  let mins_inds = map2 (\off len -> off + len - 1) offsets (map i64.i32 shp)
  let mins = map (\i -> if i == -1 then 0 else scanned_mins[i]) mins_inds

  -- Calculating maxs
  let scanned_maxs = sgmscan f32.max f32.lowest (map i32.i64 flagArray) input
  let maxs_inds = map2 (\off len -> off + len - 1) offsets (map i64.i32 shp)
  let maxs = map (\i -> if i == -1 then 0 else scanned_maxs[i]) maxs_inds
  
  -- Calculating means
  let means = map2 (\min max -> (min + max) / 2) mins maxs --|> opague 
  
  -- Calculate ks
  let ks = map (\ x -> i32.f32 (f32.floor ((f32.i32 x) / 2f32))) shp

  let A = copy input
  let med_vals = rankSearchBatch means ks shp (map i32.i64 II1) A
  let max_1st = seg_red (f32.max) (-f32.inf) (map (bool.i64)flagArray) input (map(\x -> i64.i32 x) shp)

  let index_arr = map3(\x i j -> if x == max_1st[i] then j else -1) input (map (\x -> x-1)II1) (iota m)
  let change = scatter (copy input) index_arr (replicate m (-f32.inf))
  let max_2nd = seg_red (f32.max) (-f32.inf) (map (bool.i64)flagArray) change (map(\x -> i64.i32 x) shp)
  let med_vals_update = map (\x -> if med_vals[x] == max_1st[x] && max_2nd[x] != (-f32.inf) then max_2nd[x] else med_vals[x]) (iota (length med_vals))

  in med_vals_update
