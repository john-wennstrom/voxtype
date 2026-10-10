use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::BTreeMap;
use voxtype_lexicon::{
    build::{self, Entry},
    PhoneticIndex, Scratch, SearchOpts,
};

fn retrieval(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("retrieval");
    group.sample_size(20);
    for size in [10_000, 100_000, 1_000_000] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bench.vxl");
        let entries: Vec<_> = (0..size)
            .map(|position| {
                let mut seed = position as u64 + 1;
                let phones = (0..16)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 32) % 39) as u8
                    })
                    .collect();
                Entry {
                    canonical: format!("term{position}"),
                    phones,
                    source: 0,
                    weight: 1.0,
                }
            })
            .collect();
        let exact = entries[0].phones.clone();
        let mut one_edit = exact.clone();
        one_edit[8] = (one_edit[8] + 1) % 39;
        let unrelated = vec![19, 36, 33, 6, 19, 2, 22, 30, 27, 24, 20, 13, 16, 17, 25, 9];
        build::write(entries, BTreeMap::new(), &path, 0).unwrap();
        let index = PhoneticIndex::open(&path).unwrap();
        let mut scratch = Scratch::new(&index);
        let mut candidates = Vec::with_capacity(8);
        for (name, phones) in [
            ("exact", exact),
            ("one_edit", one_edit),
            ("unrelated", unrelated),
        ] {
            group.bench_with_input(BenchmarkId::new(name, size), &phones, |bench, phones| {
                bench.iter(|| {
                    index.search(
                        black_box(phones),
                        &SearchOpts::default(),
                        &mut scratch,
                        &mut candidates,
                    );
                    black_box(&candidates);
                });
            });
        }
        group.bench_with_input(BenchmarkId::new("open", size), &size, |bench, _| {
            bench.iter(|| PhoneticIndex::open(black_box(&path)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, retrieval);
criterion_main!(benches);
