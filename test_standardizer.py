import numpy as np
from standardizer import ZScoreStandardizer


def test_basic_standardization():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    result = standardizer.fit_transform(data)
    assert result.shape == data.shape
    np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(result.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-10)
    print("✓ test_basic_standardization passed")


def test_list_input():
    data = [[10, 20], [30, 40], [50, 60]]
    standardizer = ZScoreStandardizer()
    result = standardizer.fit_transform(data)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)
    print("✓ test_list_input passed")


def test_1d_input():
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    result = standardizer.fit_transform(data)
    assert result.shape == (5, 1)
    np.testing.assert_allclose(result.mean(axis=0), [0.0], atol=1e-10)
    np.testing.assert_allclose(result.std(axis=0, ddof=0), [1.0], atol=1e-10)
    print("✓ test_1d_input passed")


def test_inverse_transform():
    data = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    transformed = standardizer.fit_transform(data)
    recovered = standardizer.inverse_transform(transformed)
    np.testing.assert_allclose(recovered, data, atol=1e-10)
    print("✓ test_inverse_transform passed")


def test_fit_then_transform():
    train = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    test = np.array([[2, 3]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    standardizer.fit(train)
    result = standardizer.transform(test)
    expected = (test - standardizer.mean_) / standardizer.scale_
    np.testing.assert_allclose(result, expected, atol=1e-10)
    print("✓ test_fit_then_transform passed")


def test_get_params():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    standardizer.fit(data)
    params = standardizer.get_params()
    assert "mean" in params
    assert "scale" in params
    np.testing.assert_allclose(params["mean"], data.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(params["scale"], data.std(axis=0, ddof=0), atol=1e-10)
    print("✓ test_get_params passed")


def test_nan_rejection():
    data = np.array([[1, np.nan], [3, 4]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    try:
        standardizer.fit_transform(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "NaN" in str(e)
    print("✓ test_nan_rejection passed")


def test_inf_rejection():
    data = np.array([[1, np.inf], [3, 4]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    try:
        standardizer.fit_transform(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "infinite" in str(e)
    print("✓ test_inf_rejection passed")


def test_without_mean():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    standardizer = ZScoreStandardizer(with_mean=False)
    result = standardizer.fit_transform(data)
    np.testing.assert_allclose(result.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-10)
    print("✓ test_without_mean passed")


def test_without_std():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    standardizer = ZScoreStandardizer(with_std=False)
    result = standardizer.fit_transform(data)
    np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)
    print("✓ test_without_std passed")


def test_zero_std_raises():
    data = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    try:
        standardizer.fit_transform(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zero standard deviation" in str(e)
        assert "0" in str(e)
    print("✓ test_zero_std_raises passed")


def test_zero_std_with_std_false():
    data = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float64)
    standardizer = ZScoreStandardizer(with_std=False)
    result = standardizer.fit_transform(data)
    np.testing.assert_allclose(result[:, 0], [0.0, 0.0, 0.0], atol=1e-10)
    assert len(standardizer.zero_std_features_) == 0
    print("✓ test_zero_std_with_std_false passed")


def test_zero_std_single_column():
    data = np.array([[3], [3], [3], [3]], dtype=np.float64)
    standardizer = ZScoreStandardizer()
    try:
        standardizer.fit_transform(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zero standard deviation" in str(e)
    print("✓ test_zero_std_single_column passed")


def test_invalid_method_raises():
    try:
        ZScoreStandardizer(method="unknown")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown method" in str(e)
    print("✓ test_invalid_method_raises passed")


def test_minmax_basic():
    data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64)
    standardizer = ZScoreStandardizer(method="minmax")
    result = standardizer.fit_transform(data)
    assert result.shape == data.shape
    np.testing.assert_allclose(result.min(axis=0), [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(result.max(axis=0), [1.0, 1.0], atol=1e-10)
    print("✓ test_minmax_basic passed")


def test_minmax_custom_range():
    data = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)
    standardizer = ZScoreStandardizer(method="minmax", feature_range=(-1, 1))
    result = standardizer.fit_transform(data)
    np.testing.assert_allclose(result.min(axis=0), [-1.0, -1.0], atol=1e-10)
    np.testing.assert_allclose(result.max(axis=0), [1.0, 1.0], atol=1e-10)
    print("✓ test_minmax_custom_range passed")


def test_minmax_invalid_feature_range():
    try:
        ZScoreStandardizer(method="minmax", feature_range=(1, 0))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "feature_range" in str(e)
    print("✓ test_minmax_invalid_feature_range passed")


def test_minmax_constant_column_raises():
    data = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float64)
    standardizer = ZScoreStandardizer(method="minmax")
    try:
        standardizer.fit_transform(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zero range" in str(e)
    print("✓ test_minmax_constant_column_raises passed")


def test_minmax_inverse_transform():
    data = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)
    standardizer = ZScoreStandardizer(method="minmax")
    transformed = standardizer.fit_transform(data)
    recovered = standardizer.inverse_transform(transformed)
    np.testing.assert_allclose(recovered, data, atol=1e-10)
    print("✓ test_minmax_inverse_transform passed")


def test_minmax_list_input():
    data = [[10, 20], [30, 40], [50, 60]]
    standardizer = ZScoreStandardizer(method="minmax")
    result = standardizer.fit_transform(data)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result.min(axis=0), [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(result.max(axis=0), [1.0, 1.0], atol=1e-10)
    print("✓ test_minmax_list_input passed")


def test_minmax_1d_input():
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    standardizer = ZScoreStandardizer(method="minmax")
    result = standardizer.fit_transform(data)
    assert result.shape == (5, 1)
    np.testing.assert_allclose(result.min(axis=0), [0.0], atol=1e-10)
    np.testing.assert_allclose(result.max(axis=0), [1.0], atol=1e-10)
    print("✓ test_minmax_1d_input passed")


def test_minmax_fit_then_transform():
    train = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    test = np.array([[4, 3]], dtype=np.float64)
    standardizer = ZScoreStandardizer(method="minmax")
    standardizer.fit(train)
    result = standardizer.transform(test)
    data_min = standardizer.mean_
    data_range = standardizer.scale_
    expected = (test - data_min) / data_range
    np.testing.assert_allclose(result, expected, atol=1e-10)
    print("✓ test_minmax_fit_then_transform passed")


def test_minmax_get_params():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    standardizer = ZScoreStandardizer(method="minmax")
    standardizer.fit(data)
    params = standardizer.get_params()
    assert params["method"] == "minmax"
    assert "data_min" in params
    assert "data_range" in params
    np.testing.assert_allclose(params["data_min"], data.min(axis=0), atol=1e-10)
    np.testing.assert_allclose(params["data_range"], np.ptp(data, axis=0), atol=1e-10)
    print("✓ test_minmax_get_params passed")


if __name__ == "__main__":
    test_basic_standardization()
    test_list_input()
    test_1d_input()
    test_inverse_transform()
    test_fit_then_transform()
    test_get_params()
    test_nan_rejection()
    test_inf_rejection()
    test_without_mean()
    test_without_std()
    test_zero_std_raises()
    test_zero_std_with_std_false()
    test_zero_std_single_column()
    test_invalid_method_raises()
    test_minmax_basic()
    test_minmax_custom_range()
    test_minmax_invalid_feature_range()
    test_minmax_constant_column_raises()
    test_minmax_inverse_transform()
    test_minmax_list_input()
    test_minmax_1d_input()
    test_minmax_fit_then_transform()
    test_minmax_get_params()
    print("\nAll tests passed!")
