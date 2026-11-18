(() => {
const $ = id => document.getElementById(id);
const qs = new URLSearchParams(location.search);
let saved = null;
try { saved = localStorage.getItem("rl_api"); } catch {}
let base = (qs.get("api") || saved || "http://localhost:8000").replace(/\/+$/, "");
let tripId = null, mode = "hmm";
$("apiUrl").value = base;

const map = L.map("map", {preferCanvas: true}).setView([54.7, 56.1], 11);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 19, attribution: "© OpenStreetMap"}).addTo(map);
const G = {net: L.layerGroup(), gps: L.layerGroup().addTo(map), snap: L.layerGroup(), route: L.layerGroup().addTo(map)};

const pct = v => v == null ? "–" : (v * 100).toFixed(1) + "%";
const busy = t => { $("busy").hidden = !t; if (t) $("busyTxt").textContent = t; };
const notice = t => { $("msgs").hidden = !t; $("msgs").textContent = t || ""; };
async function api(path, opts) {
  const r = await fetch(base + path, opts);
  if (!r.ok) {
    let d = r.statusText;
    try { d = (await r.json()).detail || d; } catch {}
    throw new Error(`${r.status}: ${typeof d === "string" ? d : JSON.stringify(d)}`);
  }
  return r.json();
}
const color = s => s == null ? "#8b97a8" : s < .5 ? "#e5484d" : s < .8 ? "#f5a524" : "#2fb36b";

async function connect() {
  const st = $("status");
  $("trips").innerHTML = ""; $("run").disabled = true; tripId = null; notice("");
  try {
    const h = await api("/health");
    st.className = "pill " + (h.status === "ok" ? "ok" : "warn");
    st.textContent = h.status === "ok" ? `online · ${h.model_name}` : "degraded";
    notice(h.messages.join(" "));
    const t = await api("/trips");
    $("partition").textContent = `(${t.partition})`;
    t.trips.forEach(tr => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="id">${tr.trip_id}</span><span class="meta">${tr.n_observations} pts · ${(tr.duration_s / 60).toFixed(1)} min</span>`;
      li.onclick = () => selectTrip(tr, li);
      $("trips").appendChild(li);
    });
    if (!t.trips.length && !h.messages.length) notice("No trips served by the API.");
  } catch (e) {
    st.className = "pill off"; st.textContent = "offline";
    notice(`Cannot reach API at ${base} (${e.message})`);
  }
}
$("apiSave").onclick = () => {
  base = $("apiUrl").value.trim().replace(/\/+$/, "");
  try { localStorage.setItem("rl_api", base); } catch {}
  connect();
};

function clear() { Object.values(G).forEach(g => g.clearLayers()); $("results").hidden = true; $("legend").hidden = true; }
function selectTrip(tr, li) {
  document.querySelectorAll(".trips li").forEach(x => x.classList.remove("sel"));
  li.classList.add("sel");
  tripId = tr.trip_id; $("run").disabled = false; clear();
  map.fitBounds([[tr.bbox[0], tr.bbox[1]], [tr.bbox[2], tr.bbox[3]]]);
  if ($("lNet").checked) loadNetwork();
}
async function loadNetwork() {
  if (!tripId) return;
  const id = tripId; G.net.clearLayers();
  try {
    const n = await api(`/trips/${id}/network`);
    if (id !== tripId) return;
    n.arcs.forEach(a => L.polyline(a, {color: "#7c8aa0", weight: 1.5, opacity: .55, interactive: false}).addTo(G.net));
  } catch (e) { notice(e.message); }
}

document.querySelectorAll("#mode button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#mode button").forEach(x => x.classList.remove("on"));
  b.classList.add("on"); mode = b.dataset.v;
});
for (const [id, k] of [["lNet", "net"], ["lGps", "gps"], ["lSnap", "snap"], ["lRoute", "route"]])
  $(id).onchange = e => {
    if (e.target.checked) { G[k].addTo(map); if (k === "net" && !G.net.getLayers().length) loadNetwork(); }
    else map.removeLayer(G[k]);
  };

$("run").onclick = async () => {
  busy("Reconstructing route…"); $("run").disabled = true;
  try {
    const r = await api("/reconstruct", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({trip_id: tripId, mode, compare: $("compare").checked}),
    });
    draw(r); show(r);
  } catch (e) { alert("Reconstruction failed\n" + e.message); }
  busy(null); $("run").disabled = false;
};

function draw(r) {
  ["gps", "snap", "route"].forEach(k => G[k].clearLayers());
  r.route_polylines.forEach(p => L.polyline(p, {color: "#4c8dff", weight: 5, opacity: .9}).addTo(G.route));
  r.observations.forEach(o => {
    L.circleMarker([o.lat, o.lon], {radius: 3, color: color(o.score), weight: 1, fillOpacity: .8})
      .bindTooltip(`#${o.index} · arc ${o.arc_id ?? "none"} · score ${o.score == null ? "–" : o.score.toFixed(3)}`).addTo(G.gps);
    if (o.snapped) {
      L.circleMarker(o.snapped, {radius: 2, color: "#a78bfa", weight: 1, fillOpacity: 1}).addTo(G.snap);
      L.polyline([[o.lat, o.lon], o.snapped], {color: "#a78bfa", weight: 1, opacity: .4, interactive: false}).addTo(G.snap);
    }
  });
  $("legend").hidden = false;
  const all = r.route_polylines.flat();
  if (all.length) map.fitBounds(L.latLngBounds(all).pad(.1));
}

function show(r) {
  const m = r.metrics;
  $("results").hidden = false;
  const card = (l, v) => `<div class="card"><span>${l}</span><b>${pct(v)}</b><div class="bar"><i style="width:${(v || 0) * 100}%"></i></div></div>`;
  $("cards").innerHTML = m
    ? card("Top-1 accuracy", m.top1_accuracy) + card("Route continuity", m.route_continuity) + card("Sequence F1", m.seq_f1) + card("Candidate miss rate", m.candidate_miss_rate)
    : `<div class="card" style="grid-column:1/3"><span>No reference route on the server – metrics unavailable.</span></div>`;
  const c = r.comparison;
  if (c) {
    const rows = [["Top-1", "top1_accuracy", 1], ["Continuity", "route_continuity", 1], ["Seq F1", "seq_f1", 1], ["Edit sim.", "edit_similarity", 1], ["Miss rate", "candidate_miss_rate", -1]];
    const names = Object.keys(c);
    $("cmp").innerHTML = `<tr><th></th>${names.map(n => `<th>${n === "hmm" ? "HMM" : "Classifier"}</th>`).join("")}</tr>` +
      rows.map(([l, k, dir]) => {
        const vals = names.map(n => c[n] ? c[n][k] : null);
        const real = vals.filter(v => v != null);
        const best = real.length ? (dir > 0 ? Math.max(...real) : Math.min(...real)) : null;
        const tie = real.length > 1 && real.every(v => v === real[0]);
        return `<tr><td>${l}</td>${vals.map(v => `<td class="${v != null && v === best && !tie ? "best" : ""}">${pct(v)}</td>`).join("")}</tr>`;
      }).join("");
  } else $("cmp").innerHTML = "";
  $("extra").textContent =
    `model: ${r.model_name} (${r.emission_source})\n` +
    `observations: ${r.n_observations_used}/${r.n_observations_raw} used, ${r.n_matched} matched, ${r.n_segment_breaks} breaks\n` +
    `route arcs: ${r.route_arc_ids.length}\n\n` +
    Object.entries(r.timings_ms).map(([k, v]) => `${k}: ${Math.round(v)} ms`).join("\n") + "\n\n" +
    Object.entries(r.cleaning_report).map(([k, v]) => `${k}: ${v}`).join("\n");
}
connect();
})();
