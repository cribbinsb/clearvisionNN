"""The data pipeline: raw video in, clean targets and degraded inputs out (DESIGN.md section 9).

Stages talk only through the records in `clearvisionnn.pipeline.records`, stored in
the work directory; a stage never calls into another stage.
"""
