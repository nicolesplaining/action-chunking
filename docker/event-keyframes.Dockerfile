FROM action-chunking-libero-client

COPY constraints/event-keyframes.txt /tmp/event-keyframes.txt
RUN uv pip install --python /.venv/bin/python -r /tmp/event-keyframes.txt
