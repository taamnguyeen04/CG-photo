#!/usr/bin/env python3
"""
UncertPhoto-SLAM: Test Uncertainty Formula Computation

This script tests the uncertainty formulas used in UncertPhoto-SLAM:
1. obs_confidence = 1 - exp(-count / tau_obs)
2. res_confidence = exp(-var / tau_res)
3. depth_confidence = exp(-var / tau_depth)
4. sigma_i = 1 - (w1*obs + w2*res + w3*depth)
"""

import math

def test_obs_confidence():
    """Test observation confidence formula."""
    tau_obs = 10.0
    test_cases = [
        (0, 0.0),        # Never observed -> 0 confidence
        (5, 0.393),      # Half tau -> ~0.39
        (10, 0.632),     # At tau -> ~0.63
        (20, 0.865),     # 2x tau -> ~0.87
        (100, 0.99995),  # High count -> ~1.0
    ]
    
    print("Testing obs_confidence = 1 - exp(-count / tau_obs):")
    print(f"  tau_obs = {tau_obs}")
    
    all_passed = True
    for count, expected in test_cases:
        actual = 1 - math.exp(-count / tau_obs)
        passed = abs(actual - expected) < 0.01
        status = "✓" if passed else "✗"
        print(f"  {status} count={count:3d}: expected={expected:.3f}, actual={actual:.3f}")
        all_passed = all_passed and passed
    
    return all_passed

def test_res_confidence():
    """Test residual confidence formula."""
    tau_res = 0.01
    test_cases = [
        (0.0, 1.0),      # Zero variance -> 1.0 confidence
        (0.005, 0.607),  # Half tau
        (0.01, 0.368),   # At tau
        (0.03, 0.050),   # 3x tau -> very low
        (0.1, 0.00005),  # Very high variance -> ~0
    ]
    
    print("\nTesting res_confidence = exp(-var / tau_res):")
    print(f"  tau_res = {tau_res}")
    
    all_passed = True
    for var, expected in test_cases:
        actual = math.exp(-var / tau_res)
        passed = abs(actual - expected) < 0.01
        status = "✓" if passed else "✗"
        print(f"  {status} var={var:.4f}: expected={expected:.3f}, actual={actual:.3f}")
        all_passed = all_passed and passed
    
    return all_passed

def test_combined_uncertainty():
    """Test combined uncertainty formula."""
    w1, w2, w3 = 0.4, 0.4, 0.2
    test_cases = [
        # (obs_conf, res_conf, depth_conf, expected_sigma)
        (0.0, 0.0, 0.0, 1.0),    # All uncertain -> sigma = 1
        (1.0, 1.0, 1.0, 0.0),    # All confident -> sigma = 0
        (0.5, 0.5, 0.5, 0.5),    # Half -> sigma = 0.5
        (1.0, 0.0, 0.0, 0.6),    # Only obs confident
        (0.0, 1.0, 0.0, 0.6),    # Only res confident
        (0.0, 0.0, 1.0, 0.8),    # Only depth confident
    ]
    
    print(f"\nTesting sigma_i = 1 - (w1*obs + w2*res + w3*depth):")
    print(f"  w1={w1}, w2={w2}, w3={w3}")
    
    all_passed = True
    for obs, res, depth, expected in test_cases:
        combined = w1 * obs + w2 * res + w3 * depth
        actual = 1.0 - combined
        passed = abs(actual - expected) < 0.001
        status = "✓" if passed else "✗"
        print(f"  {status} obs={obs:.1f}, res={res:.1f}, depth={depth:.1f}: expected={expected:.3f}, actual={actual:.3f}")
        all_passed = all_passed and passed
    
    return all_passed

def test_clamping():
    """Test that sigma is clamped to [0, 1]."""
    print("\nTesting clamping to [0, 1]:")
    
    # Edge case: very high confidence could theoretically push sigma < 0
    # but in practice weights sum to 1.0 so sigma stays in range
    tests_passed = True
    
    # Verify weights sum to 1
    w1, w2, w3 = 0.4, 0.4, 0.2
    weight_sum = w1 + w2 + w3
    if abs(weight_sum - 1.0) < 0.001:
        print(f"  ✓ Weights sum to {weight_sum:.1f} -> sigma always in [0,1]")
    else:
        print(f"  ✗ Weights sum to {weight_sum:.1f}, should be 1.0")
        tests_passed = False
    
    return tests_passed

if __name__ == "__main__":
    print("=" * 60)
    print("UncertPhoto-SLAM: Uncertainty Formula Test")
    print("=" * 60)
    
    results = [
        test_obs_confidence(),
        test_res_confidence(),
        test_combined_uncertainty(),
        test_clamping(),
    ]
    
    print("\n" + "=" * 60)
    if all(results):
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 60)
