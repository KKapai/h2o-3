package water.api.schemas3;

import hex.SegmentModelsBuilder;
import water.api.API;

public class SegmentModelsParametersV3 extends SchemaV3<SegmentModelsBuilder.SegmentModelsParameters, SegmentModelsParametersV3> {

  @API(help = "FIXME")
  public KeyV3.FrameKeyV3 segments;

  @API(help = "FIXME")
  public String[] segment_columns;

}
