const state = {
  listings: [],
  allListings: [],
  markers: new Map(),
  activeId: null,
  lastChatListingIds: null,
};

const map = L.map("map", {
  zoomControl: false,
}).setView([15.8, 106.4], 6);

L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);

const markerLayer = L.layerGroup().addTo(map);

const elements = {
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatMessages: document.getElementById("chatMessages"),
  sendButton: document.getElementById("sendButton"),
  suggestedPrompts: document.getElementById("suggestedPrompts"),
  ragStatus: document.getElementById("ragStatus"),
  searchInput: document.getElementById("searchInput"),
  listingType: document.getElementById("listingType"),
  provinceSelect: document.getElementById("provinceSelect"),
  listingList: document.getElementById("listingList"),
  evidenceSummary: document.getElementById("evidenceSummary"),
  totalCount: document.getElementById("totalCount"),
  saleCount: document.getElementById("saleCount"),
  rentCount: document.getElementById("rentCount"),
  fitMapButton: document.getElementById("fitMapButton"),
  mapTitle: document.getElementById("mapTitle"),
};

function money(vnd) {
  if (!vnd) return "Thỏa thuận";
  if (vnd >= 1_000_000_000) return `${(vnd / 1_000_000_000).toFixed(1).replace(".0", "")} tỷ`;
  return `${Math.round(vnd / 1_000_000)} triệu`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function markerIcon(item) {
  const label = item.listing_type === "cho-thue" ? "T" : "B";
  const rentClass = item.listing_type === "cho-thue" ? " rent" : "";
  return L.divIcon({
    className: "",
    html: `<span class="pin${rentClass}">${label}</span>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

function popupHtml(item) {
  const sourceLink = item.url
    ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">Open listing</a>`
    : "";
  return `
    <div class="popup">
      <h3>${escapeHtml(item.title)}</h3>
      <div class="meta-row">
        <span class="pill ${item.listing_type === "ban" ? "sale" : ""}">${item.listing_type === "ban" ? "For sale" : "For rent"}</span>
        <span class="pill">${escapeHtml(money(item.price_vnd))}</span>
        ${item.area_m2 ? `<span class="pill">${escapeHtml(item.area_m2)} m²</span>` : ""}
      </div>
      <p>${escapeHtml(item.address)}</p>
      <p>${item.geo_precision === "approximate" ? "Approximate pin from address text." : "Exact geocoded pin."}</p>
      ${sourceLink}
    </div>
  `;
}

function appendMessage(role, text, isError = false) {
  const message = document.createElement("article");
  message.className = `message ${role}${isError ? " error" : ""}`;
  message.innerHTML = `
    <div class="message-label">${role === "user" ? "You" : "Assistant"}</div>
    <p>${escapeHtml(text)}</p>
  `;
  elements.chatMessages.appendChild(message);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  return message;
}

function setLoading(isLoading) {
  elements.sendButton.disabled = isLoading;
  elements.sendButton.textContent = isLoading ? "..." : "Send";
}

function renderStats(stats) {
  elements.totalCount.textContent = stats.count ?? 0;
  elements.saleCount.textContent = stats.sale_count ?? 0;
  elements.rentCount.textContent = stats.rent_count ?? 0;
  elements.evidenceSummary.textContent = `${stats.count ?? 0} listings in current evidence set`;
  elements.mapTitle.textContent = `${stats.count ?? 0} mapped listings`;
}

function statsFor(items) {
  return {
    count: items.length,
    sale_count: items.filter((item) => item.listing_type === "ban").length,
    rent_count: items.filter((item) => item.listing_type === "cho-thue").length,
  };
}

function renderMap(items) {
  markerLayer.clearLayers();
  state.markers.clear();

  for (const item of items) {
    const marker = L.marker([item.lat, item.lng], { icon: markerIcon(item) })
      .bindPopup(popupHtml(item))
      .on("click", () => setActive(item.id, false));
    marker.addTo(markerLayer);
    state.markers.set(item.id, marker);
  }
}

function renderList(items) {
  elements.listingList.innerHTML = items
    .map((item) => {
      const img = item.image
        ? `<img src="${escapeHtml(item.image)}" alt="" loading="lazy" />`
        : `<img alt="" />`;
      const area = item.area_m2 ? `<span class="pill">${escapeHtml(item.area_m2)} m²</span>` : "";
      const source = item.url
        ? `<a class="source-link" href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">Open listing</a>`
        : "";
      return `
        <article class="listing-card ${state.activeId === item.id ? "active" : ""}" data-id="${escapeHtml(item.id)}">
          ${img}
          <div class="listing-body">
            <h3>${escapeHtml(item.title)}</h3>
            <div class="meta-row">
              <span class="pill ${item.listing_type === "ban" ? "sale" : ""}">${item.listing_type === "ban" ? "Sale" : "Rent"}</span>
              <span class="pill">${escapeHtml(money(item.price_vnd))}</span>
              ${area}
            </div>
            <p class="address">${escapeHtml(item.address)}</p>
            ${source}
          </div>
        </article>
      `;
    })
    .join("");
}

function renderEvidence(items, shouldFit = true) {
  state.listings = items;
  renderStats(statsFor(items));
  renderMap(items);
  renderList(items);
  if (shouldFit) fitMap();
}

function fitMap() {
  const markers = [...state.markers.values()];
  if (!markers.length) return;
  const bounds = L.latLngBounds(markers.map((marker) => marker.getLatLng()));
  map.fitBounds(bounds.pad(0.18), { maxZoom: 13 });
}

function setActive(id, openPopup = true) {
  state.activeId = id;
  renderList(state.listings);
  const card = elements.listingList.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (card) card.scrollIntoView({ block: "nearest" });
  const marker = state.markers.get(id);
  if (marker && openPopup) {
    marker.openPopup();
    map.panTo(marker.getLatLng(), { animate: true });
  }
}

function applyLocalFilters(baseItems = null) {
  const source = baseItems || (state.lastChatListingIds
    ? state.allListings.filter((item) => state.lastChatListingIds.has(item.id))
    : state.allListings);
  const q = elements.searchInput.value.trim().toLowerCase();
  const type = elements.listingType.value;
  const province = elements.provinceSelect.value.toLowerCase();

  let items = source;
  if (type) items = items.filter((item) => item.listing_type === type);
  if (province) items = items.filter((item) => item.province.toLowerCase().includes(province));
  if (q) {
    items = items.filter((item) =>
      [item.title, item.address, item.project, item.property_type, item.district]
        .join(" ")
        .toLowerCase()
        .includes(q)
    );
  }
  renderEvidence(items);
}

async function loadListings() {
  const res = await fetch("/api/listings");
  const payload = await res.json();
  state.allListings = payload.items;
  state.lastChatListingIds = null;
  renderEvidence(payload.items);
}

async function hydrateProvinces() {
  const res = await fetch("/api/listings");
  const payload = await res.json();
  const provinces = [...new Set(payload.items.map((item) => item.province).filter(Boolean))].sort();
  for (const province of provinces) {
    const option = document.createElement("option");
    option.value = province;
    option.textContent = province;
    elements.provinceSelect.appendChild(option);
  }
}

function listingsFromChat(payload) {
  if (Array.isArray(payload.listings) && payload.listings.length) {
    const ids = new Set(payload.listings.map((item) => item.id));
    state.lastChatListingIds = ids;
    return state.allListings.filter((item) => ids.has(item.id));
  }
  state.lastChatListingIds = null;
  return state.allListings;
}

async function submitChat(message) {
  const text = message.trim();
  if (!text) return;

  appendMessage("user", text);
  elements.chatInput.value = "";
  setLoading(true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const payload = await res.json();
    appendMessage("assistant", payload.answer || "Không có phản hồi.", Boolean(payload.error));
    elements.ragStatus.textContent = payload.error ? "Fallback" : "RAG";
    elements.ragStatus.classList.toggle("error", Boolean(payload.error));
    applyLocalFilters(listingsFromChat(payload));
  } catch (error) {
    appendMessage("assistant", `Không gọi được /api/chat: ${error}`, true);
    elements.ragStatus.textContent = "Offline";
    elements.ragStatus.classList.add("error");
  } finally {
    setLoading(false);
  }
}

let searchTimer = null;
elements.searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => applyLocalFilters(), 180);
});
elements.listingType.addEventListener("change", () => applyLocalFilters());
elements.provinceSelect.addEventListener("change", () => applyLocalFilters());
elements.fitMapButton.addEventListener("click", fitMap);
elements.listingList.addEventListener("click", (event) => {
  if (event.target.closest("a")) return;
  const card = event.target.closest(".listing-card");
  if (card) setActive(card.dataset.id);
});
elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitChat(elements.chatInput.value);
});
elements.chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitChat(elements.chatInput.value);
  }
});
elements.suggestedPrompts.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  submitChat(button.textContent);
});

hydrateProvinces().then(loadListings);
