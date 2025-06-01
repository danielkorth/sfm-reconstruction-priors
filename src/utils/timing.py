import time
from contextlib import contextmanager
from typing import Optional


@contextmanager
def timing(description: str, verbose: bool = True, indent: int = 0):
    """Context manager for timing code blocks.

    Args:
        description: Description of the operation being timed
        verbose: Whether to print timing information
        indent: Indentation level for prettier nested timing

    Example:
        >>> with timing("Loading data"):
        >>>     data = load_data()
        >>>
        >>> # For nested timing
        >>> with timing("Processing pipeline"):
        >>>     with timing("Feature extraction", indent=4):
        >>>         features = extract_features(data)
        >>>     with timing("Model training", indent=4):
        >>>         model = train_model(features)
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time

        if verbose:
            indent_str = " " * indent

            # Format time appropriately based on magnitude
            if elapsed_time < 0.001:
                time_str = f"{elapsed_time * 1000000:.2f} μs"
            elif elapsed_time < 1.0:
                time_str = f"{elapsed_time * 1000:.2f} ms"
            else:
                # Format time differently based on duration
                if elapsed_time < 60:
                    time_str = f"{elapsed_time:.3f} s"
                elif elapsed_time < 3600:
                    minutes, seconds = divmod(elapsed_time, 60)
                    time_str = f"{int(minutes)}m {seconds:.2f}s"
                else:
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

            print(f"Timing:{indent_str}⏱️ {description}: {time_str}")


class TimingStats:
    """Class to collect and report timing statistics across multiple runs.

    Example:
        >>> timing_stats = TimingStats()
        >>>
        >>> # Collect timing data
        >>> for dataset in datasets:
        >>>     with timing_stats.time("Data loading"):
        >>>         data = load_data(dataset)
        >>>
        >>>     with timing_stats.time("Processing"):
        >>>         process_data(data)
        >>>
        >>> # Print summary
        >>> timing_stats.print_summary()
    """

    def __init__(self):
        self.timings = {}

    @contextmanager
    def time(self, description: str, verbose: bool = True, indent: int = 0):
        """Same as the timing context manager but also collects statistics."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time

            if description not in self.timings:
                self.timings[description] = []
            self.timings[description].append(elapsed_time)

            if verbose:
                indent_str = " " * indent

                # Format time appropriately based on magnitude
                if elapsed_time < 0.001:
                    time_str = f"{elapsed_time * 1000000:.2f} μs"
                elif elapsed_time < 1.0:
                    time_str = f"{elapsed_time * 1000:.2f} ms"
                else:
                    # Format time differently based on duration
                    if elapsed_time < 60:
                        time_str = f"{elapsed_time:.3f} s"
                    elif elapsed_time < 3600:
                        minutes, seconds = divmod(elapsed_time, 60)
                        time_str = f"{int(minutes)}m {seconds:.2f}s"
                    else:
                        hours, remainder = divmod(elapsed_time, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

                print(f"{indent_str}⏱️ {description}: {time_str}")

    def print_summary(self):
        """Print summary statistics of all collected timings."""
        if not self.timings:
            print("No timing data collected")
            return

        print("\n===== Timing Summary =====")
        for description, times in self.timings.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            print(f"\n{description}:")
            print(f"  Calls: {len(times)}")
            print(f"  Total: {self._format_time(sum(times))}")
            print(f"  Average: {self._format_time(avg_time)}")
            print(f"  Min: {self._format_time(min_time)}")
            print(f"  Max: {self._format_time(max_time)}")

    @staticmethod
    def _format_time(elapsed_time):
        if elapsed_time < 0.001:
            return f"{elapsed_time * 1000000:.2f} μs"
        elif elapsed_time < 1.0:
            return f"{elapsed_time * 1000:.2f} ms"
        elif elapsed_time < 60:
            return f"{elapsed_time:.3f} s"
        elif elapsed_time < 3600:
            minutes, seconds = divmod(elapsed_time, 60)
            return f"{int(minutes)}m {seconds:.2f}s"
        else:
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
