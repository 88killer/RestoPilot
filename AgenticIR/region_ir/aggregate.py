"""Source-level, seed-aware summaries for experiment JSONL records.

Input rows: source_id, scenario, method, seed, status ('ok'/'failed'), metrics,
seconds, calls. Optional dataset defaults to 'core'; different datasets never pool.
A failed row should contain metrics of the declared fallback output. Missing values
are counted and never zero-imputed. Duplicate method/seed/source/scenario rows fail.

For each metric: average valid scenarios within each source and training seed;
report per-seed source-bootstrap CIs. Across seeds, average source means only for
sources represented in every seed, then bootstrap sources. Seed-mean standard
deviation is separate, never used as additional independent image observations.
Paired comparison uses the same source, seed, and scenarios in both methods.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

METRICS = ('psnr', 'ssim', 'lpips', 'normal_mae', 'normal_change', 'boundary_mae', 'boundary_gradient_mae')


def metric_values(values):
    result = {name: values.get(name) for name in METRICS}
    for parent in ('region_psnr', 'region_ssim'):
        for name, value in values.get(parent, {}).items():
            result[parent + '.' + name] = value
    segmentation = values.get('segmentation', {})
    if segmentation:
        result['segmentation.normal_false_positive_rate'] = segmentation.get('normal_false_positive_rate')
        for name, scores in segmentation.get('classes', {}).items():
            for score in ('iou', 'dice'):
                result[f'segmentation.{name}.{score}'] = scores.get(score)
    return result


def numeric(value):
    if value == 'infinity':
        return float('inf')
    if value == '-infinity':
        return float('-inf')
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'Invalid metric value: {value!r}')
    if not np.isfinite(value):
        raise ValueError('Use explicit infinity strings; NaN metrics are not allowed')
    return float(value)


def encoded(value):
    if value is None or np.isnan(value):
        return None
    if np.isposinf(value):
        return 'infinity'
    if np.isneginf(value):
        return '-infinity'
    return float(value)


def mean(values):
    if not values:
        return None
    positive, negative = any(np.isposinf(v) for v in values), any(np.isneginf(v) for v in values)
    if positive and negative:
        return None
    if positive:
        return float('inf')
    if negative:
        return float('-inf')
    return float(np.mean(values))


def source_summary(source_values, rng, bootstrap_samples, confidence):
    values = list(source_values.values())
    summary = dict(mean=encoded(mean(values)), source_count=len(values),
                   infinite_source_count=sum(not np.isfinite(value) for value in values),
                   ci=None, ci_method='percentile bootstrap of source means')
    if not values:
        summary['ci_unavailable_reason'] = 'no valid source means'
    elif not all(np.isfinite(value) for value in values):
        summary['ci_unavailable_reason'] = 'non-finite source mean'
    elif len(values) < 2:
        summary['ci_unavailable_reason'] = 'fewer than two independent sources'
    else:
        array = np.asarray(values)
        # Bound memory for large external datasets without changing the resampling unit.
        estimates = np.empty(bootstrap_samples)
        for start in range(0, bootstrap_samples, 128):
            count = min(128, bootstrap_samples - start)
            estimates[start:start + count] = array[rng.integers(len(array), size=(count, len(array)))].mean(axis=1)
        alpha = (1 - confidence) / 2
        summary['ci'] = np.quantile(estimates, [alpha, 1 - alpha]).tolist()
    return summary


def seed_summary(values_by_seed, rng, bootstrap_samples, confidence):
    per_seed = {seed: source_summary(values, rng, bootstrap_samples, confidence)
                for seed, values in sorted(values_by_seed.items())}
    all_sources = set().union(*(set(v) for v in values_by_seed.values()))
    complete = set.intersection(*(set(v) for v in values_by_seed.values())) if values_by_seed else set()
    combined = {}
    for source in sorted(complete):
        value = mean([v[source] for v in values_by_seed.values()])
        if value is not None:
            combined[source] = value
    seed_means = [mean(list(values.values())) for values in values_by_seed.values()]
    finite_seed_means = [value for value in seed_means if value is not None and np.isfinite(value)]
    return dict(per_seed=per_seed, across_seeds=source_summary(combined, rng, bootstrap_samples, confidence),
                training_seed_count=len(values_by_seed),
                source_count_missing_one_or_more_seeds=len(all_sources - complete),
                seed_mean_std=float(np.std(finite_seed_means, ddof=1)) if len(finite_seed_means) == len(seed_means) and len(seed_means) > 1 else None,
                seed_mean_std_note='Across training seed means, separately from source-bootstrap CI')


def cost_summary(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'Invalid nonnegative resource measurement: {key}')
        values.append(value)
    return dict(count=len(values), missing_count=len(rows) - len(values),
                mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                p95=float(np.percentile(values, 95)) if values else None,
                total=float(np.sum(values)) if values else None)


def aggregate_rows(rows, baseline=None, bootstrap_samples=2000, confidence=.95, seed=42):
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid bootstrap protocol')
    datasets = defaultdict(list)
    seen = set()
    for original in rows:
        row = dict(original)
        for key in ('source_id', 'scenario', 'method', 'seed', 'status'):
            if key not in row:
                raise ValueError(f'Missing experiment field: {key}')
        if row['status'] not in ('ok', 'failed'):
            raise ValueError('status must be ok or failed')
        row['dataset'] = str(row.get('dataset', 'core'))
        for key in ('source_id', 'scenario', 'method', 'seed'):
            row[key] = str(row[key])
        key = tuple(row[k] for k in ('dataset', 'method', 'seed', 'source_id', 'scenario'))
        if key in seen:
            raise ValueError(f'Duplicate experiment record: {key}')
        seen.add(key)
        row['_metrics'] = {key: numeric(value) for key, value in metric_values(row.get('metrics', {})).items()}
        datasets[row['dataset']].append(row)
    rng = np.random.default_rng(seed)
    output = dict(protocol=dict(unit='source image', scenario_reduction='mean within source and training seed',
                                seed_reduction='mean across seeds for sources present in every seed',
                                confidence=confidence, bootstrap_samples=bootstrap_samples, bootstrap_seed=seed,
                                paired_sign='method minus baseline', failed_rows='include measured fallback metrics and all resource costs'),
                  datasets={})
    for dataset, dataset_rows in sorted(datasets.items()):
        methods = defaultdict(list)
        for row in dataset_rows:
            methods[row['method']].append(row)
        report = dict(methods={}, paired={})
        for name, group in sorted(methods.items()):
            summary = dict(rows=len(group), completed=sum(r['status'] == 'ok' for r in group),
                           failed=sum(r['status'] == 'failed' for r in group),
                           completion_rate=sum(r['status'] == 'ok' for r in group) / len(group),
                           seconds=cost_summary(group, 'seconds'), calls=cost_summary(group, 'calls'),
                           metrics={})
            seeds = sorted({r['seed'] for r in group})
            keys = sorted(set().union(*(r['_metrics'].keys() for r in group)))
            for key in keys:
                buckets = {seed_id: defaultdict(list) for seed_id in seeds}
                valid = 0
                for row in group:
                    value = row['_metrics'].get(key)
                    if value is not None:
                        buckets[row['seed']][row['source_id']].append(value)
                        valid += 1
                means = {seed_id: {source: value for source, scores in sources.items()
                                   if (value := mean(scores)) is not None}
                         for seed_id, sources in buckets.items()}
                metric = seed_summary(means, rng, bootstrap_samples, confidence)
                metric.update(valid_rows=valid, missing_rows=len(group) - valid,
                              failed_rows_with_metric=sum(r['status'] == 'failed' and r['_metrics'].get(key) is not None for r in group))
                summary['metrics'][key] = metric
            report['methods'][name] = summary
        if baseline is not None:
            if baseline not in methods:
                raise ValueError(f'Baseline {baseline!r} absent from dataset {dataset!r}')
            base = {(r['seed'], r['source_id'], r['scenario']): r for r in methods[baseline]}
            for name, group in sorted(methods.items()):
                if name == baseline:
                    continue
                other = {(r['seed'], r['source_id'], r['scenario']): r for r in group}
                matched_keys = set(base) & set(other)
                # Do not compare source means computed over different scene collections.
                eligible = set()
                source_seeds = {(k[0], k[1]) for k in set(base) | set(other)}
                for seed_id, source in source_seeds:
                    a = {k[2] for k in base if k[:2] == (seed_id, source)}
                    b = {k[2] for k in other if k[:2] == (seed_id, source)}
                    if a == b and a:
                        eligible.add((seed_id, source))
                paired = dict(baseline=baseline, matched_rows=len(matched_keys),
                              unmatched_rows=len(set(base) ^ set(other)),
                              excluded_source_seed_groups=len(source_seeds - eligible), metrics={})
                keys = sorted(set().union(*(r['_metrics'].keys() for r in group + methods[baseline])))
                common_seeds = sorted({k[0] for k in base} & {k[0] for k in other})
                for key in keys:
                    buckets = {seed_id: defaultdict(list) for seed_id in common_seeds}
                    matched_metric_rows = 0
                    for row_key in sorted(matched_keys):
                        if row_key[:2] not in eligible:
                            continue
                        a, b = base[row_key]['_metrics'].get(key), other[row_key]['_metrics'].get(key)
                        if a is None or b is None or (np.isinf(a) and a == b):
                            continue
                        buckets[row_key[0]][row_key[1]].append(b - a)
                        matched_metric_rows += 1
                    means = {seed_id: {source: value for source, values in sources.items()
                                       if (value := mean(values)) is not None}
                             for seed_id, sources in buckets.items()}
                    value = seed_summary(means, rng, bootstrap_samples, confidence)
                    value['paired_valid_rows'] = matched_metric_rows
                    paired['metrics'][key] = value
                report['paired'][name] = paired
        output['datasets'][dataset] = report
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('records', nargs='+', help='Experiment JSONL files')
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = []
    for path in args.records:
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip())
    result = aggregate_rows(rows, args.baseline, args.bootstrap_samples, seed=args.seed)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
