def log2 x = (loop (y,c) = (x,0i32) while y > 1i32 do (y >> 1, c+1)).1

def iota32 n = (0..1..<i32.i64 n) :> [n]i32

def imap  as f = map f as
def imap2 as bs f = map2 f as bs

def ones [q] 't (_xs: [q]t) = replicate q 1i32

-- Functions from DPP notes
def mkFlagArray 't [m] 
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

-- Please implement the function below, which is supposed to 
-- be the lifted version of `partition2` function given above.
-- Arguments:
--   `(shp: [m]i32, arr: [n]t)` is the flat-representation of
--            the irregular 2-dim (input) array to be partitioned;
--            `shp` is its shape, and `arr` is its flat data; 
--   `condsL` is an irregular 2-dim array of booleans, which has
--            the same shape (`shp`) and flat-length (`n`) as the
--            input to-be-partitioned array.  
-- The result is a tuple:
--    the first element is an array of split points of size `m`,
--       i.e., the index in each segment where the `false` elements
--       start.
--    the second element is the flat-representation of the partitioned result:
--       the first element should simply be `shp` (redundant)
--       the second element should be the flat-data of the partitioned result.
--let partition2L 't [n] [m]
--                -- the shape of condsL is also shp
--                (condsL: [n]bool) (dummy: t)
--                (shp: [m]i64, arr: [n]t) :
--                ([m]i32, ([m]i64, [n]t)) =
--  let begs   = scan (+) 0 shp
--  let flags  =  (  iota m
--                |> map i32.i64
--                |> map (+1)
--                |> mkFlagArray shp 0i32
--                ) :> [n]i32
--
--  let outinds= sgmSumInt flags <| map (\f -> if f==0 then 0 else f-1) flags
--
--  let tflgsL = map (\c -> if c then 1i32 else 0i32) condsL
--  let fflgsL = map (\b -> 1 - b) tflgsL
--
--  let indsTL= sgmSumInt flags tflgsL
--  let tmpL  = sgmSumInt flags fflgsL
--
--  -- let lst = indsT[n-1]
--  let lstL   = map2 (\s b -> if s==0 then -1 else #[unsafe] indsTL[b-1]
--                    ) shp begs
--
--  -- let indsF = map (+lst) tmp
--  let indsFL = map2 (\t sgmind-> t + #[unsafe] lstL[sgmind]) tmpL outinds
--
--  let indsL = map4(\c indT indF sgmind->
--                        let offs = if sgmind > 0 then #[unsafe] begs[sgmind-1] else 0i64
--                        in  if c then offs + (i64.i32 indT) - 1
--                                 else offs + (i64.i32 indF) - 1
--                  ) condsL indsTL indsFL outinds
--
--  let fltarrL = scatter (replicate n dummy) indsL arr
--  in  (lstL, (shp,fltarrL))

def partition3L2 't [n] [p]
        (mask : [n]bool) -- mask[i] == True => associated predicate holds on elem i              [f,t,f,t,f,t,f,t,f,t,f,t]
        (shp_flag_arr: [n]i32)                                                                -- [1,0,0,1,0,0,0,1,0,0,0,0]
        (scan_shp: [p]i32)                                                                    --     [0,0,3,3,7,12]
        (shp : [p]i32, flat_arr : [n]t) -- representation of an irregular array of array      -- shp:[0,0,3,0,4,5]
        (dummy : t)
      : ([n]t, [p]i32) = -- result: the flat array reorganized & splitting point of each segment

  let ffs = map (\f -> if f then 0 else 1) mask                                                   -- [1,0,1,0,1,0,1,0,1,0,1,0]
  let tfs = map (\f -> if f then 1 else 0) mask                                               -- [0,1,0,1,0,1,0,1,0,1,0,1]
  let isT = sgmscan (+) 0 shp_flag_arr tfs                                                    -- [0,1,1,1,1,2,2,1,1,2,2,3]
  let splits = map2 (\s off -> if s == 0 then 0 else isT[off-1]) shp scan_shp :> [p]i32       -- [1,2,3]

  -- Since you have many different segments you want to know the indicies of the current segment.
  --  You therefore add the exclusive scaned shape array elem to the start of each segment of tfs
  
  let exc_scan_shp = ( [0i64] ++ (map (\i -> i64.i32 i) scan_shp[:(p - 1)]) ) :> [p]i64           -- [0,0,0,3,3,7]

  let isT_segments = 
      let tfs_add_shp      = map (\ind -> tfs[ind] + (i32.i64 ind)) exc_scan_shp                  --[0,0,0,4,4,8]
      let tfs_with_seg_val = scatter (copy tfs) (exc_scan_shp) tfs_add_shp                        --[0,1,0,4,0,1,0,8,0,1,0,1]
      in sgmscan (+) 0 shp_flag_arr tfs_with_seg_val                                              --[0,1,1,4,4,5,5,8,8,9,9,10]

  let isT_segments_last_elem = map2 (\ind s -> if s == 0 then -1 else isT_segments[ind-1]) 
                                                                      scan_shp shp :> [p]i32      -- [-1,-1,1,-1,5,10]   
  let isT_ind = map2 (\off s -> if s == -1 then -1 else off) exc_scan_shp isT_segments_last_elem  -- [-1,-1,0,-1,3,7]
  let isF_segments =
      let ffs_add_Ts       = map2 (\ind t_val -> ffs[ind] + t_val) 
                                            exc_scan_shp isT_segments_last_elem                   -- [0,0,2,2,5,10]
      let ffs_with_seg_val = scatter (copy ffs) (isT_ind) ffs_add_Ts                              -- [2,0,1,5, 1,0,1,10,1,0,1,0] 
      in sgmscan (+) 0 shp_flag_arr ffs_with_seg_val

  let inds = map3 (\c iT iF -> if c    then i64.i32(iT -1)
                                       else i64.i32(iF -1)
                  ) mask isT_segments isF_segments
  let r =  scatter (replicate n dummy) inds flat_arr
  in (r, splits) 


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


def computeMedianWithRankK [m][n] (ass: [m][n]f32) =
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
    in medians

-- ==
-- compiled input {
--  [false,true,false,true,false,true,false,true,false,true,false,true]
--  [3,4,5]
--  [1,2,3,4,5,6,7,8,9,10,11,12]  
-- }
-- output {
-- [2,1,3,4,6,5,7,8,10,12,9,11]
-- [1,2,3] 
--}
--let main [m] [n] (mask: [n]bool) (shp: [m]i32) (f_arr : [n]i32) =
--  let scan_shp = scan (+) 0 shp
--  let shp_flag_arr = (mkFlagArray shp (replicate m 1i32)) :> [n]i32
--  let (new_flat_arr, splits) =  partition3L2 mask shp_flag_arr scan_shp (shp, f_arr)
--  in (new_flat_arr, splits)