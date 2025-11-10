const state = {
  snapshots: [],
  mode: "cumulative",
  playing: false,
  timer: null,
  cumulativeIndex: 0,
  sliceStart: 0,
  sliceEnd: 1,
  stepIndex: 0,
};

const elements = {
  plot: document.getElementById("plot"),
  modeSelect: document.getElementById("mode"),
  playButton: document.getElementById("playButton"),
  reloadButton: document.getElementById("reloadButton"),
  status: document.getElementById("status"),
  cumulativePanel: document.getElementById("cumulativePanel"),
  cumulativeSlider: document.getElementById("cumulativeSlider"),
  cumulativeInfo: document.getElementById("cumulativeInfo"),
  slicePanel: document.getElementById("slicePanel"),
  sliceStart: document.getElementById("sliceStart"),
  sliceEnd: document.getElementById("sliceEnd"),
  sliceInfo: document.getElementById("sliceInfo"),
  stepPanel: document.getElementById("stepPanel"),
  stepSlider: document.getElementById("stepSlider"),
  stepInfo: document.getElementById("stepInfo"),
};

init();

function init() {
  attachListeners();
  refreshData();
}

function attachListeners() {
  elements.modeSelect.addEventListener("change", () => {
    state.mode = elements.modeSelect.value;
    stopPlayback();
    updatePanels();
    renderActiveMode();
  });

  elements.reloadButton.addEventListener("click", () => {
    stopPlayback();
    refreshData();
  });

  elements.playButton.addEventListener("click", () => {
    if (state.playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  elements.cumulativeSlider.addEventListener("input", (event) => {
    state.cumulativeIndex = Number(event.target.value);
    renderCumulative();
  });

  elements.sliceStart.addEventListener("input", () => {
    const startValue = Number(elements.sliceStart.value);
    if (startValue >= state.sliceEnd) {
      state.sliceEnd = Math.min(startValue + 1, state.snapshots.length - 1);
      elements.sliceEnd.value = state.sliceEnd;
    }
    state.sliceStart = startValue;
    enforceSliceBounds();
    renderSlice();
  });

  elements.sliceEnd.addEventListener("input", () => {
    const endValue = Number(elements.sliceEnd.value);
    if (endValue <= state.sliceStart) {
      state.sliceStart = Math.max(endValue - 1, 0);
      elements.sliceStart.value = state.sliceStart;
    }
    state.sliceEnd = endValue;
    enforceSliceBounds();
    renderSlice();
  });

  elements.stepSlider.addEventListener("input", (event) => {
    state.stepIndex = Number(event.target.value);
    renderStep();
  });
}

async function refreshData() {
  setStatus("Loading…");
  try {
    const response = await fetch("/data", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const payload = await response.json();
    state.snapshots = Array.isArray(payload.snapshots) ? payload.snapshots : [];
    configureControls();
    updatePanels();
    renderActiveMode();
    setStatus(`Loaded ${state.snapshots.length} snapshot(s).`);
  } catch (error) {
    setStatus(error.message || "Failed to load data.", true);
    Plotly.react(elements.plot, [], {
      title: "No data",
      xaxis: { visible: false },
      yaxis: { visible: false },
      annotations: [
        { text: "Unable to load histogram data.", x: 0.5, y: 0.5, showarrow: false, xref: "paper", yref: "paper" },
      ],
    });
  }
}

function configureControls() {
  const count = state.snapshots.length;
  const lastIndex = Math.max(count - 1, 0);

  // Cumulative slider
  elements.cumulativeSlider.max = String(lastIndex);
  elements.cumulativeSlider.disabled = count === 0;
  state.cumulativeIndex = Math.min(state.cumulativeIndex, lastIndex);
  elements.cumulativeSlider.value = String(state.cumulativeIndex);

  // Slice sliders
  const hasSlice = count >= 2;
  elements.sliceStart.disabled = !hasSlice;
  elements.sliceEnd.disabled = !hasSlice;
  elements.sliceStart.max = String(Math.max(count - 2, 0));
  elements.sliceEnd.max = String(Math.max(count - 1, 1));
  state.sliceStart = clamp(state.sliceStart, 0, Math.max(count - 2, 0));
  state.sliceEnd = clamp(state.sliceEnd, state.sliceStart + 1, Math.max(count - 1, 1));
  elements.sliceStart.value = String(state.sliceStart);
  elements.sliceEnd.value = String(state.sliceEnd);
  enforceSliceBounds();

  // Step slider
  const hasStep = count >= 2;
  elements.stepSlider.disabled = !hasStep;
  elements.stepSlider.max = String(Math.max(count - 2, 0));
  state.stepIndex = clamp(state.stepIndex, 0, Math.max(count - 2, 0));
  elements.stepSlider.value = String(state.stepIndex);

  // Play button availability
  const canPlay = (state.mode === "cumulative" && count > 1) || (state.mode === "step" && count > 1);
  if (!canPlay) {
    stopPlayback();
  }
  elements.playButton.disabled = !canPlay;
}

function updatePanels() {
  elements.cumulativePanel.classList.toggle("hidden", state.mode !== "cumulative");
  elements.slicePanel.classList.toggle("hidden", state.mode !== "slice");
  elements.stepPanel.classList.toggle("hidden", state.mode !== "step");
  elements.playButton.hidden = state.mode === "slice";
  configureControls();
}

function enforceSliceBounds() {
  if (state.snapshots.length < 2) {
    state.sliceStart = 0;
    state.sliceEnd = 1;
    return;
  }

  const maxStart = Math.max(state.snapshots.length - 2, 0);
  state.sliceStart = clamp(state.sliceStart, 0, maxStart);
  state.sliceEnd = clamp(state.sliceEnd, state.sliceStart + 1, state.snapshots.length - 1);
  elements.sliceStart.max = String(maxStart);
  elements.sliceEnd.max = String(state.snapshots.length - 1);
  elements.sliceStart.value = String(state.sliceStart);
  elements.sliceEnd.value = String(state.sliceEnd);
}

function renderActiveMode() {
  if (state.mode === "slice") {
    renderSlice();
  } else if (state.mode === "step") {
    renderStep();
  } else {
    renderCumulative();
  }
}

function renderCumulative() {
  const snapshots = state.snapshots;
  if (!snapshots.length) {
    elements.cumulativeInfo.textContent = "No snapshots available.";
    drawEmptyPlot("Awaiting snapshots…");
    return;
  }

  const snap = snapshots[state.cumulativeIndex];
  if (!snap) {
    drawEmptyPlot("Snapshot index is out of range.");
    return;
  }

  const bins = sortBins(snap.bins);
  const total = bins.reduce((sum, bucket) => sum + bucket.count, 0);
  elements.cumulativeInfo.textContent = snapshotLabel(snap) + ` · total ${total}`;
  drawPlot(bins, `Snapshot ${snap.index}`, snap.label ? snap.label : undefined);
}

function renderSlice() {
  if (state.snapshots.length < 2) {
    elements.sliceInfo.textContent = "Need at least two snapshots to compute a slice.";
    drawEmptyPlot("Slice mode unavailable.");
    return;
  }

  const earlier = state.snapshots[state.sliceStart];
  const later = state.snapshots[state.sliceEnd];
  const bins = diffBins(later, earlier);
  const delta = bins.reduce((sum, bucket) => sum + bucket.count, 0);
  elements.sliceInfo.textContent = `Diff from snapshot ${earlier.index} → ${later.index} · total Δ ${delta}`;
  drawPlot(bins, `Slice ${earlier.index} → ${later.index}`, describeLabels(earlier, later));
}

function renderStep() {
  if (state.snapshots.length < 2) {
    elements.stepInfo.textContent = "Need at least two snapshots to compute steps.";
    drawEmptyPlot("Step mode unavailable.");
    return;
  }

  const start = state.snapshots[state.stepIndex];
  const end = state.snapshots[state.stepIndex + 1];
  const bins = diffBins(end, start);
  const delta = bins.reduce((sum, bucket) => sum + bucket.count, 0);
  elements.stepInfo.textContent = `Step ${start.index} → ${end.index} · total Δ ${delta}`;
  drawPlot(bins, `Step ${start.index} → ${end.index}`, describeLabels(start, end));
}

function drawPlot(bins, title, subtitle) {
  const trace = {
    type: "bar",
    x: bins.map((bucket) => bucket.label),
    y: bins.map((bucket) => bucket.count),
    marker: { color: "#3f51b5" },
    hovertemplate: "%{x}<br>count=%{y}<extra></extra>",
  };

  const layout = {
    title: subtitle ? `${title} — ${subtitle}` : title,
    margin: { t: 60, r: 20, b: 80, l: 60 },
    xaxis: { title: "Bucket", automargin: true },
    yaxis: { title: "Count", rangemode: "tozero" },
  };

  Plotly.react(elements.plot, [trace], layout, { responsive: true, displaylogo: false });
}

function drawEmptyPlot(message) {
  Plotly.react(
    elements.plot,
    [],
    {
      title: message,
      xaxis: { visible: false },
      yaxis: { visible: false },
    },
    { responsive: true, displaylogo: false },
  );
}

function diffBins(later, earlier) {
  if (!later || !earlier) {
    return [];
  }

  const map = new Map();
  for (const bucket of later.bins || []) {
    map.set(bucket.key, { ...bucket });
  }

  for (const bucket of earlier.bins || []) {
    const existing = map.get(bucket.key) || {
      ...bucket,
      count: 0,
    };
    existing.count -= bucket.count;
    map.set(bucket.key, existing);
  }

  return sortBins(Array.from(map.values()));
}

function sortBins(bins) {
  return [...bins].sort((a, b) => {
    const startA = valueForSort(a.start, a.start_label);
    const startB = valueForSort(b.start, b.start_label);
    if (startA === startB) {
      return a.end_label.localeCompare(b.end_label);
    }
    return startA - startB;
  });
}

function valueForSort(value, label) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (label === "-inf") {
    return Number.NEGATIVE_INFINITY;
  }
  if (label === "inf") {
    return Number.POSITIVE_INFINITY;
  }
  return 0;
}

function snapshotLabel(snapshot) {
  return snapshot.label ? `Snapshot ${snapshot.index} (${snapshot.label})` : `Snapshot ${snapshot.index}`;
}

function describeLabels(start, end) {
  const startLabel = start.label ? `${start.index} (${start.label})` : `${start.index}`;
  const endLabel = end.label ? `${end.index} (${end.label})` : `${end.index}`;
  return `${startLabel} → ${endLabel}`;
}

function startPlayback() {
  if (state.playing) {
    return;
  }
  const count = state.snapshots.length;
  if (count < 2) {
    return;
  }

  state.playing = true;
  elements.playButton.textContent = "Pause";
  state.timer = setInterval(() => {
    if (state.mode === "cumulative") {
      if (state.cumulativeIndex >= count - 1) {
        stopPlayback();
        return;
      }
      state.cumulativeIndex += 1;
      elements.cumulativeSlider.value = String(state.cumulativeIndex);
      renderCumulative();
    } else if (state.mode === "step") {
      const maxStep = count - 2;
      if (state.stepIndex >= maxStep) {
        stopPlayback();
        return;
      }
      state.stepIndex += 1;
      elements.stepSlider.value = String(state.stepIndex);
      renderStep();
    } else {
      stopPlayback();
    }
  }, 800);
}

function stopPlayback() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  state.playing = false;
  elements.playButton.textContent = "Play";
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", Boolean(isError));
}

function clamp(value, min, max) {
  if (Number.isNaN(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}
