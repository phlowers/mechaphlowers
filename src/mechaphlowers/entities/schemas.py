import numpy as np
import pandera as pa
from pandera.typing import DataFrame, Series


class SectionArrayInput(pa.DataFrameModel):
    """Schema for the data expected for a dataframe used to instantiate a SectionArray.

    Each row describes a support and the following span (except the last row which "only" describes the last support).

    Notes:
        Line angles are expressed in degrees.

        insulator_length should be zero for the first and last supports, since for now mechaphlowers
        ignores them when computing the state of a span or section.
        Taking them into account might be implemented later.

        span_length should be zero or numpy.nan for the last row.
    """

    name: Series[str]
    suspension: Series[bool]
    conductor_attachment_altitude: Series[float] = pa.Field(coerce=True)
    crossarm_length: Series[float] = pa.Field(coerce=True)
    line_angle: Series[float] = pa.Field(coerce=True)
    insulator_length: Series[float] = pa.Field(coerce=True)
    span_length: Series[float] = pa.Field(nullable=True, coerce=True)

    @pa.dataframe_check(
        description="""Though tension supports also have insulators,
        for now we ignore them when computing the state of a span or section.
        Taking them into account might be implemented later.
        For now, set the insulator length to 0 for tension supports to suppress this error."""
    )
    def insulator_length_is_zero_if_not_suspension(cls, df: DataFrame) -> Series[bool]:
        return (df["suspension"] | (df["insulator_length"] == 0)).pipe(Series[bool])

    @pa.dataframe_check(
        description="""Each row in the dataframe contains information about a support
        and the span next to it, except the last support which doesn't have a "next" span.
        So, specifying a span_length in the last row doesn't make any sense.
        Please set span_length to "not a number" (numpy.nan) to suppress this error.""",
    )
    def no_span_length_for_last_row(cls, df: DataFrame) -> bool:
        return df.tail(1)["span_length"].isin([0, np.nan]).all()