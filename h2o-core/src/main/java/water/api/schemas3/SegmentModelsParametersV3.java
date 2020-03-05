package water.api.schemas3;

import hex.SegmentModelsBuilder;
import water.api.API;

public class SegmentModelsParametersV3 extends SchemaV3<SegmentModelsBuilder.SegmentModelsParameters, SegmentModelsParametersV3> {

  @API(help = "Enumeration of all segments for which to build models for")
  public KeyV3.FrameKeyV3 segments;

  @API(help = "List of columns to segment-by, models will be built for all segments in the data")
  public String[] segment_columns;

}
