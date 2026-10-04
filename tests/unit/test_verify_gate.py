from tools.verify import pytest_worker_args


def test_serial_default_emits_explicit_zero_workers() -> None:
    assert pytest_worker_args(
        force_serial=False,
        env_workers=None,
        available_memory_gb=16.0,
        test_file_count=5,
        cpu_count=8,
        xdist_available=True,
    ) == ["-p", "no:cacheprovider", "-n", "0"]


def test_missing_xdist_omits_worker_flag() -> None:
    result = pytest_worker_args(
        force_serial=False,
        env_workers="4",
        available_memory_gb=16.0,
        test_file_count=5,
        cpu_count=8,
        xdist_available=False,
    )
    assert result == ["-p", "no:cacheprovider"]
    assert "-n" not in result


def test_missing_xdist_omits_worker_flag_even_when_forced_serial() -> None:
    result = pytest_worker_args(
        force_serial=True,
        env_workers="4",
        available_memory_gb=16.0,
        test_file_count=5,
        cpu_count=8,
        xdist_available=False,
    )
    assert result == ["-p", "no:cacheprovider"]


def test_opt_in_parallel_is_bounded_by_cpus_and_files() -> None:
    assert pytest_worker_args(
        force_serial=False,
        env_workers="8",
        available_memory_gb=16.0,
        test_file_count=3,
        cpu_count=4,
        xdist_available=True,
    ) == ["-p", "no:cacheprovider", "-n", "3"]


def test_unknown_cpu_count_falls_back_to_two() -> None:
    result = pytest_worker_args(
        force_serial=False,
        env_workers="8",
        available_memory_gb=16.0,
        test_file_count=10,
        cpu_count=None,
        xdist_available=True,
    )
    assert result == ["-p", "no:cacheprovider", "-n", "2"]


def test_low_memory_forces_serial_and_boundary_is_inclusive() -> None:
    assert pytest_worker_args(
        force_serial=False,
        env_workers="4",
        available_memory_gb=1.99,
        test_file_count=5,
        cpu_count=8,
        xdist_available=True,
    ) == ["-p", "no:cacheprovider", "-n", "0"]
    assert pytest_worker_args(
        force_serial=False,
        env_workers="4",
        available_memory_gb=2.0,
        test_file_count=5,
        cpu_count=8,
        xdist_available=True,
    ) == ["-p", "no:cacheprovider", "-n", "4"]


def test_force_serial_flag_overrides_env() -> None:
    assert pytest_worker_args(
        force_serial=True,
        env_workers="4",
        available_memory_gb=16.0,
        test_file_count=5,
        cpu_count=8,
        xdist_available=True,
    ) == ["-p", "no:cacheprovider", "-n", "0"]


def test_malformed_or_single_worker_env_stays_serial() -> None:
    for env_workers in ("", "1", "0", "abc", "-2", "2.5"):
        assert pytest_worker_args(
            force_serial=False,
            env_workers=env_workers,
            available_memory_gb=16.0,
            test_file_count=5,
            cpu_count=8,
            xdist_available=True,
        ) == ["-p", "no:cacheprovider", "-n", "0"]


def test_cache_provider_always_disabled() -> None:
    combos: list[tuple[str | None, float, int, int | None, bool]] = [
        (None, 16.0, 5, 8, True),
        ("4", 16.0, 5, 8, True),
        ("4", 16.0, 5, 8, False),
        ("8", 1.0, 3, None, True),
    ]
    for env_workers, mem, files, cpus, xdist in combos:
        result = pytest_worker_args(
            force_serial=False,
            env_workers=env_workers,
            available_memory_gb=mem,
            test_file_count=files,
            cpu_count=cpus,
            xdist_available=xdist,
        )
        assert result[:2] == ["-p", "no:cacheprovider"]
