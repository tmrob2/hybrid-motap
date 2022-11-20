use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hybrid::tests::integration::{test_blis_threads, test_cb_threads, test_rayon_threads, test_explicit_threads_size2, test_explicit_threads_size4, test_explicit_threads_size8, test_rayon_spmv_c, test_rayon_spmv_sprs, csparse_mv, sprs_mv};

pub fn criterion_benchmark(c: &mut Criterion) {
    //c.bench_function("blis exps 1", |b| b.iter(|| test_blis_threads()));
    //c.bench_function("blis exps 1CPU", |b| b.iter(|| test_cb_threads(black_box(1))));
    c.bench_function("blis exps 2CPU", |b| b.iter(|| test_cb_threads(black_box(2))));
    c.bench_function("blis exps 2 explicit threads", |b| b.iter(|| test_explicit_threads_size2()));
    c.bench_function("blis exps 4 explicit threads", |b| b.iter(|| test_explicit_threads_size4()));
    c.bench_function("blis exps 8 explicit threads", |b| b.iter(|| test_explicit_threads_size8()));
    //c.bench_function("blis exps 4CPU", |b| b.iter(|| test_cb_threads(black_box(4))));
    //c.bench_function("blis exps 8CPU", |b| b.iter(|| test_cb_threads(black_box(8))));
    //c.bench_function("blis exps 12CPU", |b| b.iter(|| test_cb_threads(black_box(12))));
    //c.bench_function("blis exps 16CPU", |b| b.iter(|| test_cb_threads(black_box(16))));
    c.bench_function("blis exps rayon", |b| b.iter(|| test_rayon_threads(black_box(100000))));
    //c.bench_function("blis exps rayon par iter 1000", |b| b.iter(|| test_rayon_threads(black_box(1000))));
    //c.bench_function("blis exps rayon par iter 10000", |b| b.iter(|| test_rayon_threads(black_box(10000))));
    c.bench_function("blis exps rayon sparse", |b| b.iter(|| test_rayon_spmv_c(black_box(100000))));
    c.bench_function("sprs rayon sparse", |b| b.iter(|| test_rayon_spmv_sprs(black_box(100000))));
    c.bench_function("blis sparse", |b| b.iter(|| csparse_mv()));
    c.bench_function("sprs sparse", |b| b.iter(|| sprs_mv()));

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);