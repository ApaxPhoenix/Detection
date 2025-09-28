import sys
import asyncio
import numpy as np
import numpy.core.defchararray as chars
import traceback
from typing import Dict, Optional, Union
import warnings


class Bar:
    """
    Terminal progress bar for tracking long-running operations.

    Displays completion percentage, processing speed, and estimated time
    remaining for iterative tasks. Supports additional metric display
    and integrates with async workflows.
    """

    def __init__(
        self, iterations: int, title: str = "Loading", steps: int = 40
    ) -> None:
        """
        Initialize progress bar with task parameters.

        Args:
            iterations: Total number of items to process
            title: Display label for the operation
            steps: Character width of the progress bar
        """
        # Total work to be completed
        self.iterations: int = iterations

        # Operation display name
        self.title: str = title

        # Visual bar width in characters
        self.steps: int = steps

        # Storage for additional metrics
        self.items: Dict[str, str] = {}

    async def update(self, batch: int, time: float, final: bool = False) -> None:
        """
        Refresh progress display with current completion status.

        Args:
            batch: Number of items completed so far
            time: Operation start timestamp for speed calculation
            final: Whether this is the final update (adds newline)
        """
        # Calculate elapsed processing time
        elapsed: float = np.subtract(asyncio.get_event_loop().time(), time)

        # Determine completion percentage
        percentage: float = np.divide(batch, self.iterations)

        # Calculate processing throughput (items per second)
        throughput: np.array = np.where(
            np.greater(elapsed, 0),  # Avoid division by zero
            np.floor_divide(batch, elapsed),
            0,
        )

        # Estimate remaining time based on current progress
        eta: np.array = np.where(
            np.greater(batch, 0),  # Require progress for estimation
            np.divide(
                np.multiply(
                    elapsed, np.subtract(self.iterations, batch)
                ),
                batch,
            ),
            0,  # Cannot estimate without initial progress
        )

        # Construct visual progress bar representation
        bar: str = chars.add(
            "|",
            chars.add(
                # Filled portion using hash characters
                "".join(np.repeat("#", np.ceil(np.multiply(percentage, self.steps)))),
                chars.add(
                    # Empty portion using spaces
                    "".join(
                        np.repeat(
                            " ",
                            np.subtract(
                                self.steps, np.ceil(np.multiply(percentage, self.steps))
                            ),
                        )
                    ),
                    # Progress counter display
                    f"| {batch:03d}/{self.iterations:03d}",
                ),
            ),
        )

        # Output complete progress line to terminal
        sys.stdout.write(
            chars.add(
                chars.add(
                    chars.add(
                        # Core progress information
                        f"\r{self.title}: {bar} [{np.multiply(percentage, 100):.2f}%] in {elapsed:.1f}s "
                        f"({throughput:.1f}/s, ETA: {eta:.1f}s)",
                        # Additional metrics if available
                        np.where(
                            np.greater(np.size(self.items), 0),
                            chars.add(
                                " (",
                                chars.add(
                                    ", ".join(
                                        [
                                            f"{name}: {value}"
                                            for name, value in self.items.items()
                                        ]
                                    ),
                                    ")",
                                ),
                            ),
                            "",  # No additional metrics to display
                        ),
                    ),
                    "",
                ),
                "",
            )
        )

        # Add newline for final update
        if final:
            sys.stdout.write("\n")

        # Force immediate terminal output
        sys.stdout.flush()

    async def postfix(self, **kwargs: Union[str, int, float]) -> None:
        """
        Update supplementary metrics displayed alongside progress.

        Accepts arbitrary key-value pairs for displaying additional
        information such as loss values, accuracy, or other metrics.

        Examples:
            await pbar.postfix(loss=0.5, accuracy=0.95)
            await pbar.postfix(lr=0.001, batch_size=32)
        """
        # Update metrics dictionary with new values
        self.items.update(kwargs)

    async def __aenter__(self) -> "Bar":
        """
        Enable usage as async context manager.

        Returns:
            Bar instance for use within async context block
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[traceback.TracebackException],
    ) -> None:
        """
        Handle cleanup when exiting async context manager.

        Shows completion status on normal exit or error message on exception.
        """
        if exc_type is None:
            # Normal completion - display final status
            await self.update(
                self.iterations,  # Mark all work as complete
                asyncio.get_event_loop().time(),  # Current timestamp
                final=True,  # Add newline for clean exit
            )
        else:
            # Exception occurred - display error notification
            warnings.warn(f"\n{self.title} encountered error: {exc_val}")