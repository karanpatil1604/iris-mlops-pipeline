from datetime import timedelta
import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32
from feast import ValueType

iris = Entity(name="iris", join_keys=["iris_id"], value_type=ValueType.INT64)

iris_data_source = FileSource(
    name="iris_data_source",
    path="data/iris_data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

iris_fv = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    online=True,
    source=iris_data_source,
    tags={"domain": "iris_classification"},
)

iris_feature_service_v1 = FeatureService(
    name="iris_feature_service_v1",
    features=[iris_fv],
)
