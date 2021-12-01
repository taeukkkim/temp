from .data_collator import (
    DataCollatorWithPadding,
)


DATA_COLLATOR = {
    "extractive": DataCollatorWithPadding,
}
