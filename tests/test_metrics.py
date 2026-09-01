from trimtab.metrics import SGLANG_MAP, PrometheusMetrics, parse, quantile

T1 = """# HELP sglang:generation_tokens_total x
sglang:generation_tokens_total{model_name="m"} 1000
sglang:num_running_reqs{model_name="m"} 12
sglang:time_to_first_token_seconds_bucket{le="0.1"} 10
sglang:time_to_first_token_seconds_bucket{le="0.5"} 20
sglang:time_to_first_token_seconds_bucket{le="1"} 20
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 20
sglang:time_to_first_token_seconds_count 20
sglang:time_to_first_token_seconds_sum 5
"""
T2 = """sglang:generation_tokens_total{model_name="m"} 3000
sglang:num_running_reqs{model_name="m"} 30
sglang:time_to_first_token_seconds_bucket{le="0.1"} 10
sglang:time_to_first_token_seconds_bucket{le="0.5"} 110
sglang:time_to_first_token_seconds_bucket{le="1"} 120
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 120
sglang:time_to_first_token_seconds_count 120
sglang:time_to_first_token_seconds_sum 50
"""


def test_parse_gauge_counter_histogram():
    p = parse(T1)
    assert p["sglang:num_running_reqs"]["value"] == 12
    h = p["sglang:time_to_first_token_seconds"]
    assert h["count"] == 20 and h["buckets"][0.5] == 20 and float("inf") in h["buckets"]


def test_quantile_interpolates():
    # 100 requests in window, all between 0.1s and 0.5s, none below 0.1
    d = {0.1: 0, 0.5: 100, 1.0: 100, float("inf"): 100}
    assert abs(quantile(0.5, d) - 0.3) < 1e-9
    assert abs(quantile(0.99, d) - 0.496) < 1e-9
    assert quantile(0.5, {0.1: 0, float("inf"): 0}) is None


def test_rates_and_quantiles_between_samples():
    texts = iter([T1, T2])
    m = PrometheusMetrics({"r0": "u"}, SGLANG_MAP, fetch=lambda u: next(texts))
    first = m.sample("r0")
    assert first == {"running": 12}  # no rates on first sample
    m.prev["r0"] = (m.prev["r0"][0] - 2.0, m.prev["r0"][1])  # pretend 2s passed
    second = m.sample("r0")
    assert abs(second["throughput"] - 1000) < 5  # 2000 tokens over ~2s
    assert 100 < second["p50_ttft_ms"] < 500 and second["p99_ttft_ms"] <= 1000
    assert second["ttft_requests"] == 100 and second["running"] == 30
