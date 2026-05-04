
import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/cpprandom/shuffle"
import "lib/github.com/diku-dk/complex/complex"

local module real = { open f64 }
local module rng_engine = pcg32
local module shuffle = mk_shuffle rng_engine
module c32 = mk_complex f32

-- Permutation function
let calculate_Pj [d] (point: [d]f32) (permutation : [d]i64) : *[d]f32 =
    map (\i -> point[i]) permutation

-- Rotation function
-- Later on, try to see if you can parallelize this, e.g., scan with 2x2 matmul.
let calculate_Qj [d] [d1] (point: *[d]f32) (rand_numbers: [d1]f32) : *[d]f32 =
    loop acc = point for i < d1  do
        let tmp = acc[i] --Since the second calculation needs the original value
                         -- I store it here since it is permuted in the first calculation
        let acc[i]   = ((f32.cos rand_numbers[i]) * tmp  + (f32.sin rand_numbers[i]) * acc[i+1])
        let acc[i+1] = ((-f32.sin rand_numbers[i]) * tmp + (f32.cos rand_numbers[i]) * acc[i+1])
        in  acc


-- Fourier transform functions
let mat_vec_complex [d2] (mat : [d2][d2](f32,f32)) (vec : [d2](f32,f32))  =
    map (\row -> reduce (c32.+) (c32.mk 0.0 0.0) <| map2 (\m v -> m c32.* v) row vec ) mat

let Z [d] (point: [d]f32) (d2 : i64) : [d2](f32,f32)  =
    map (\i -> c32.mk (point[i*2]) (point[(i*2) + 1])) (iota d2)

let Z_inv [d2] [d] (TZ: [d2](f32,f32))  (point : [d]f32) : [d]f32 =
    map (\i ->  if i%2==0 && i == d-1 then point[d-1]
                else if i%2 == 0  then c32.re TZ[i/2]
                else c32.im TZ[i/2]
        ) (iota (d))

let Fd [d] (point: [d]f32) (d2 : i64) : [d]f32  =
    let pZ = Z point d2
    let one_over_d2sqr = c32.mk_re (1.0 / ( (f32.sqrt (f32.i64 d2))))
    let T = map (\k ->
                    map (\l ->
                            let exponent_im = c32.exp (c32.mk_im
                                (-(2.0 * f32.pi * (f32.i64 k) * (f32.i64 l)) / (f32.i64 d2)))
                            in one_over_d2sqr c32.* exponent_im
                        ) (iota d2)
                ) (iota d2)
    let pTZ = mat_vec_complex T pZ
    in Z_inv pTZ point

let Theta [d] (point :  [d]f32) (permutations : [][]i64) (random_numbers : [][]f32) (m1 : i64) (m2 : i64) =
    let m2_PQ = loop acc = point for i < m2 do
            let pointP = calculate_Pj acc permutations[i]
            in  calculate_Qj pointP random_numbers[i]
    let d2 = d / 2
    let Fd_m2_PQ = Fd m2_PQ d2
    in loop acc = Fd_m2_PQ for i < m1 do
            let pointP = calculate_Pj acc permutations[i + m2]
            in  calculate_Qj pointP random_numbers[i + m2]

def pseudoRandomOrthogonalTransformation [n] [d] (m: i64) (t: i32) (points : [n][d]f32) : [n][d]f32 =
    let M1 =  m/m
    let M2 =  M1

    let points' = copy points

    -- RNGs for the P and Q functions
    let rng_origin_P = rng_engine.rng_from_seed [t]
    let rngs_M1M2 = rng_engine.split_rng (M1+M2) rng_origin_P
    let (_, permutations_P) = unzip <| map (\r -> shuffle.shuffle r (iota d)) rngs_M1M2

    let rng_origin_Q = rng_engine.rng_from_seed [t*10]
    let rngs_dM1M2 = rng_engine.split_rng ((M1+M2) * (d-1)) rng_origin_Q

    -- (d-1)*(m1+m2) random numbers for Q normalized and then multiplied by 2 pi
    let (_ , rand_numbers_Q)     = unzip <| map (\r -> rng_engine.rand r) rngs_dM1M2
    let rand_numbers_Q_normalized2pi = map (\num -> (f32.f64 (((f64.u32 num) / (f64.u32 4294967295)) * (2*real.pi) ))) rand_numbers_Q
    let rndQ_normalized2d = unflatten rand_numbers_Q_normalized2pi :>[(M1+M2)][(d-1)]f32  --For v.0.22.4 use M1+M2 and (d-1) as parameters to unflatten
    -- let rndQ_normalized2d = unflatten (M1+M2) (d-1) rand_numbers_Q_normalized2pi --:>[(M1+M2)][(d-1)]f32  --For v.0.22.4 use M1+M2 and (d-1) as parameters to unflatten

    in map (\p ->  Theta p  permutations_P rndQ_normalized2d M1 M2) points'


def main [n][d] (Tval: i64) (points : [n][d]f32) = --: [][n][d]f32 =
    let (rm,am,bm) =map (\t ->
            let t = i32.i64 ((t+1) * 4)
            let M1 =  d / 2
            let M2 = M1

            let points' = copy points

            -- RNGs for the P and Q functions
            let rng_origin_P = rng_engine.rng_from_seed [t]
            let rngs_M1M2 = rng_engine.split_rng (M1+M2) rng_origin_P
            let (_, permutations_P) = unzip <| map (\r -> shuffle.shuffle r (iota d)) rngs_M1M2

            let rng_origin_Q = rng_engine.rng_from_seed [t*10]
            let rngs_dM1M2 = rng_engine.split_rng ((M1+M2) * (d-1)) rng_origin_Q

            -- (d-1)*(m1+m2) random numbers for Q normalized and then multiplied by 2 pi
            let (_ , rand_numbers_Q)     = unzip <| map (\r -> rng_engine.rand r) rngs_dM1M2
            let rand_numbers_Q_normalized2pi = map (\num -> (f32.f64 (((f64.u32 num) / (f64.u32 4294967295)) * (2*real.pi) ))) rand_numbers_Q
            let rndQ_normalized2d = unflatten rand_numbers_Q_normalized2pi :>[(M1+M2)][(d-1)]f32 --For v.0.22.4 use M1+M2 and (d-1) as parameters to unflatten
            -- let rndQ_normalized2d = unflatten (M1+M2) (d-1) rand_numbers_Q_normalized2pi --:>[(M1+M2)][(d-1)]f32  --For v.0.22.4 use M1+M2 and (d-1) as parameters to unflatten

            let r = map (\p ->  Theta p  permutations_P rndQ_normalized2d M1 M2) points'
            let a = map2 (-) points[0] points[1]
            let b = map2 (-) points[0] points[1]
            in (r , a, b)
        ) (iota Tval) |> unzip3
    in (rm,am,bm)
