import sys
import asyncio
import numpy as np
import numpy.core.defchararray as chars
import traceback
from typing import Dict, Optional, Union
import warnings


class Bar:
    """
    Shows a nice progress bar in the terminal for long-running tasks.

    You know those progress bars you see when downloading files or installing
    software? This creates one of those for your Python code. It shows how
    much is done, how fast it's going, and estimates when it'll finish.
    """

    def __init__(
        self, iterations: int, title: str = "Loading", steps: int = 40
    ) -> None:
        """
        Set up a new progress bar.

        Args:
            iterations: How many things you need to process total
            title: What to call this task (like "Training" or "Processing images")
            steps: How wide to make the bar (more steps = smoother updates)
        """
        # Remember how much work we need to do total
        self.iterations: int = iterations

        # What we're calling this task
        self.title: str = title

        # How many characters wide the bar should be
        self.steps: int = steps

        # Space for extra info like "loss=0.5, accuracy=0.9"
        self.items: Dict[str, str] = {}

    async def update(self, batch: int, time: float, final: bool = False) -> None:
        """
        Update the progress bar with current status.

        Args:
            batch: How many items we've finished so far
            time: When we started (used to calculate speed and time left)
            final: Set to True on the last update to add a newline
        """
        # How long have we been working?
        elapsed: float = np.subtract(asyncio.get_event_loop().time(), time)

        # What percentage are we done?
        percentage: float = np.divide(batch, self.iterations)

        # How fast are we going? (items per second)
        throughput: np.array = np.where(
            np.greater(elapsed, 0),  # Don't divide by zero!
            np.floor_divide(batch, elapsed),
            0,
        )

        # How much time is probably left?
        eta: np.array = np.where(
            np.greater(batch, 0),  # Need some progress to estimate
            np.divide(
                np.multiply(
                    elapsed, np.subtract(self.iterations, batch)
                ),
                batch,
            ),
            0,  # Can't estimate if we haven't started yet
        )

        # Build the actual progress bar: |####    | 025/100
        bar: str = chars.add(
            "|",
            chars.add(
                # Fill in the completed part with # symbols
                "".join(np.repeat("#", np.ceil(np.multiply(percentage, self.steps)))),
                chars.add(
                    # Fill the rest with spaces
                    "".join(
                        np.repeat(
                            " ",
                            np.subtract(
                                self.steps, np.ceil(np.multiply(percentage, self.steps))
                            ),
                        )
                    ),
                    # Add the counter at the end
                    f"| {batch:03d}/{self.iterations:03d}",
                ),
            ),
        )

        # Put together the whole line with all the info
        sys.stdout.write(
            chars.add(
                chars.add(
                    chars.add(
                        # Main progress info
                        f"\r{self.title}: {bar} [{np.multiply(percentage, 100):.2f}%] in {elapsed:.1f}s "
                        f"({throughput:.1f}/s, ETA: {eta:.1f}s)",
                        # Add any extra metrics if we have them
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
                            "",  # Nothing to add if no extra items
                        ),
                    ),
                    "",
                ),
                "",
            )
        )

        # Move to next line when we're all done
        if final:
            sys.stdout.write("\n")

        # Make sure it shows up right away
        sys.stdout.flush()

    async def postfix(self, **kwargs: Union[str, int, float]) -> None:
        """
        Add extra info to show alongside the progress bar.

        Use this to show things like loss values, accuracy, learning rate, etc.
        They'll appear in parentheses after the main progress info.

        Examples:
            await pbar.postfix(loss=0.5, accuracy=0.95)
            await pbar.postfix(lr=0.001, batch_size=32)
        """
        # Update our extra info dictionary
        self.items.update(kwargs)

    async def __aenter__(self) -> "Bar":
        """
        Let you use this with 'async with' statements.

        Like:
        async with Bar(100, "Training") as pbar:
            # do stuff and call pbar.update()
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[traceback.TracebackException],
    ) -> None:
        """
        Clean up when exiting the 'async with' block.

        If everything went well, show 100% completion.
        If something crashed, show an error message.
        """
        if exc_type is None:
            # Everything worked - show completion
            await self.update(
                self.iterations,  # Show we finished everything
                asyncio.get_event_loop().time(),  # Use current time
                final=True,  # Add the newline
            )
        else:
            # Something went wrong - let the user know
            warnings.warn(f"\n{self.title} hit an error: {exc_val}")