import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { marked } from 'marked';

const getLeaflet = () => window.L;

const escapeHtml = (value) => String(value ?? '')
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#039;');

const renderSafeMarkdown = (text) => ({
  __html: marked.parse(escapeHtml(text || ''), { breaks: true }),
});

const formatMoney = (vnd) => {
  if (!vnd) return 'Thỏa thuận';
  if (vnd >= 1_000_000_000) {
    return `${(vnd / 1_000_000_000).toFixed(1).replace('.0', '')} tỷ`;
  }
  return `${Math.round(vnd / 1_000_000)} triệu`;
};

const formatMonthlyMoney = (vnd) => {
  if (!vnd || !Number.isFinite(vnd)) return 'N/A';
  if (vnd >= 1_000_000_000) return `${(vnd / 1_000_000_000).toFixed(1)} tỷ/tháng`;
  return `${(vnd / 1_000_000).toFixed(1).replace('.0', '')} triệu/tháng`;
};

const formatPricePerM2 = (item) => {
  if (!item?.price_vnd || !item?.area_m2) return 'Chưa đủ dữ liệu';
  return `${formatMoney(item.price_vnd / item.area_m2)}/m²`;
};

const listingModeLabel = (item) => (item?.listing_type === 'cho-thue' ? 'Cho thuê' : 'Mua bán');

const geoPrecisionLabel = (value) => (value === 'exact' ? 'Tọa độ chính xác' : 'Vị trí ước lượng');

const poiLabels = {
  transit_station: 'Giao thông',
  school: 'Trường học',
  hospital: 'Y tế',
  park: 'Công viên',
};

const promptGroups = [
  {
    title: 'Lọc tin chính xác',
    icon: 'fa-filter',
    prompt: 'Tìm chung cư bán giá từ 3 đến 5 tỷ ở Quận 2 có 2 phòng ngủ',
  },
  {
    title: 'So sánh dự án',
    icon: 'fa-code-compare',
    prompt: 'So sánh căn hộ Feliz En Vista với Estella Heights về giá thuê, diện tích và pháp lý',
  },
  {
    title: 'Tiện ích xung quanh',
    icon: 'fa-location-dot',
    prompt: 'Xung quanh dự án Feliz En Vista có trường học, bệnh viện hoặc công viên nào trong bán kính 2km?',
  },
  {
    title: 'Cảm quan cư dân',
    icon: 'fa-comments',
    prompt: 'Khu vực Thạnh Mỹ Lợi Quận 2 có bị ngập nước, kẹt xe hoặc ồn vào mùa mưa không?',
  },
  {
    title: 'Tài chính & giá thị trường',
    icon: 'fa-chart-line',
    prompt: 'Tôi định mua căn hộ giá 5.5 tỷ cho 80m² ở Thạnh Mỹ Lợi. Mức giá này có hợp lý không?',
  },
  {
    title: 'Thống kê thị trường',
    icon: 'fa-chart-simple',
    prompt: 'Cho tôi biết giá trung bình và đơn giá m² của căn hộ chung cư ở Quận 2 hiện tại',
  },
];

function RecommendationCard({ item, onSelect, onCompare, selectedForCompare }) {
  return (
    <article className="recommendation-card">
      <div>
        <div className="recommendation-topline">
          <span className={`badge-tag ${item.listing_type === 'ban' ? 'sale' : 'rent'}`}>
            {listingModeLabel(item)}
          </span>
          <span className="geo-chip">{geoPrecisionLabel(item.geo_precision)}</span>
        </div>
        <h4>{item.title || 'Tin đăng chưa có tiêu đề'}</h4>
        <p><i className="fa-solid fa-location-dot" /> {item.district || item.province || item.address || 'Chưa rõ vị trí'}</p>
      </div>
      <div className="recommendation-metrics">
        <span><strong>{formatMoney(item.price_vnd)}</strong>Giá</span>
        <span><strong>{item.area_m2 ? `${item.area_m2} m²` : 'N/A'}</strong>Diện tích</span>
        <span><strong>{formatPricePerM2(item)}</strong>Đơn giá</span>
      </div>
      <p className="match-reason">
        Phù hợp để kiểm tra nhanh giá, vị trí, tiện ích xung quanh và đối chiếu nguồn RAG.
      </p>
      <div className="recommendation-actions">
        <button type="button" className="btn-api" onClick={() => onSelect(item.id)}>
          <i className="fa-solid fa-circle-info" /> Xem quyết định
        </button>
        <button type="button" className="btn-soft" onClick={() => onCompare(item)}>
          <i className={`fa-solid ${selectedForCompare ? 'fa-check' : 'fa-scale-balanced'}`} />
          {selectedForCompare ? 'Đã chọn' : 'So sánh'}
        </button>
      </div>
    </article>
  );
}

function ListingCard({ item, active, onSelect, onCompare, selectedForCompare }) {
  return (
    <article className={`listing-card decision-listing-card ${active ? 'active' : ''}`} onClick={() => onSelect(item.id)}>
      {item.image ? (
        <img src={item.image} alt={item.title || 'Bất động sản'} loading="lazy" />
      ) : (
        <div className="listing-image-placeholder"><i className="fa-solid fa-image" /></div>
      )}
      <div className="listing-info">
        <div className="listing-title-row">
          <h3>{item.title || 'Tin đăng chưa có tiêu đề'}</h3>
          <span className={`badge-tag ${item.listing_type === 'ban' ? 'sale' : 'rent'}`}>{listingModeLabel(item)}</span>
        </div>
        <div className="decision-metric-row">
          <span><strong>{formatMoney(item.price_vnd)}</strong>Giá</span>
          <span><strong>{item.area_m2 ? `${item.area_m2} m²` : 'N/A'}</strong>Diện tích</span>
          <span><strong>{formatPricePerM2(item)}</strong>Đơn giá</span>
        </div>
        <p className="listing-address"><i className="fa-solid fa-location-dot" /> {item.district || item.address || 'Chưa có địa chỉ'}</p>
        <div className="listing-quality-row">
          <span><i className="fa-solid fa-file-shield" /> {item.legal || 'Pháp lý: chưa rõ'}</span>
          <span><i className="fa-solid fa-location-crosshairs" /> {geoPrecisionLabel(item.geo_precision)}</span>
        </div>
        <div className="listing-card-actions" onClick={(event) => event.stopPropagation()}>
          <button type="button" className="btn-soft" onClick={() => onSelect(item.id)}>
            <i className="fa-solid fa-magnifying-glass-chart" /> Chi tiết
          </button>
          <button type="button" className="btn-soft" onClick={() => onCompare(item)}>
            <i className={`fa-solid ${selectedForCompare ? 'fa-check' : 'fa-scale-balanced'}`} />
            {selectedForCompare ? 'Đã chọn' : 'So sánh'}
          </button>
        </div>
      </div>
    </article>
  );
}

function CompareTable({ items, onClose, onSubmitAnalysis }) {
  const rows = [
    ['Mức giá', (item) => formatMoney(item.price_vnd)],
    ['Diện tích', (item) => (item.area_m2 ? `${item.area_m2} m²` : 'N/A')],
    ['Đơn giá', (item) => formatPricePerM2(item)],
    ['Vị trí', (item) => item.address || item.district || 'Chưa rõ'],
    ['Pháp lý', (item) => item.legal || 'Đang cập nhật'],
    ['Phòng ngủ', (item) => (item.bedrooms ? `${item.bedrooms} PN` : 'N/A')],
    ['Phòng tắm', (item) => (item.bathrooms ? `${item.bathrooms} WC` : 'N/A')],
    ['Độ tin cậy vị trí', (item) => geoPrecisionLabel(item.geo_precision)],
    ['Phù hợp nhất', (item) => (item.listing_type === 'cho-thue' ? 'Tối ưu chi phí thuê & tiện ích sống' : 'Đánh giá mua ở, đầu tư và khả năng vay')],
  ];

  return (
    <div className="compare-modal-overlay animate-fade-in" role="dialog" aria-modal="true" aria-label="So sánh bất động sản">
      <div className="compare-modal-content decision-modal">
        <div className="modal-header-row">
          <div>
            <h2>So sánh 2 bất động sản</h2>
            <p>Đối chiếu nhanh các yếu tố tác động trực tiếp đến quyết định mua hoặc thuê.</p>
          </div>
          <button type="button" className="btn-icon" onClick={onClose} aria-label="Đóng so sánh">
            <i className="fa-solid fa-times" />
          </button>
        </div>
        <div className="compare-table">
          <div className="compare-table-row compare-table-head">
            <span>Tiêu chí</span>
            <span>{items[0].title || 'BĐS 1'}</span>
            <span>{items[1].title || 'BĐS 2'}</span>
          </div>
          {rows.map(([label, getter]) => (
            <div className="compare-table-row" key={label}>
              <span>{label}</span>
              <strong>{getter(items[0])}</strong>
              <strong>{getter(items[1])}</strong>
            </div>
          ))}
        </div>
        <button type="button" className="btn-api modal-primary-action" onClick={onSubmitAnalysis}>
          <i className="fa-solid fa-robot" /> Nhờ RAG Agent phân tích sâu
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [theme, setTheme] = useState(localStorage.getItem('color-scheme') || 'auto');
  const [allListings, setAllListings] = useState([]);
  const [listings, setListings] = useState([]);
  const [lastChatListings, setLastChatListings] = useState(null);
  const [activeId, setActiveId] = useState(null);
  const [compareList, setCompareList] = useState([]);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(640);
  const [isResizing, setIsResizing] = useState(false);
  const [pendingFitMap, setPendingFitMap] = useState(false);

  const [chatMessages, setChatMessages] = useState([
    {
      role: 'assistant',
      text: 'Chào bạn! Hãy nhập nhu cầu bằng tiếng Việt tự nhiên, ví dụ: tìm chung cư yên tĩnh, gần metro, có trường học và ít ngập nước.\n\nTôi sẽ dùng RAG để truy xuất tin đăng, tiện ích xung quanh, đánh giá cộng đồng và dữ liệu tài chính liên quan.',
      payload: null,
    },
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [ragStatus, setRagStatus] = useState({ label: 'RAG', isError: false });

  const [searchQuery, setSearchQuery] = useState('');
  const [listingType, setListingType] = useState('');
  const [provinceSelect, setProvinceSelect] = useState('');
  const [minPrice, setMinPrice] = useState('');
  const [maxPrice, setMaxPrice] = useState('');
  const [minArea, setMinArea] = useState('');
  const [maxArea, setMaxArea] = useState('');
  const [bedsSelect, setBedsSelect] = useState('');
  const [sortBy, setSortBy] = useState('');
  const [ragFilterOnly, setRagFilterOnly] = useState(false);
  const [provinces, setProvinces] = useState([]);

  const [pois, setPois] = useState([]);
  const [isPoisLoading, setIsPoisLoading] = useState(false);
  const [downPaymentPct, setDownPaymentPct] = useState(30);
  const [interestRate, setInterestRate] = useState(9);
  const [loanTermYears, setLoanTermYears] = useState(20);
  const [monthlyIncome, setMonthlyIncome] = useState('');

  const mapRef = useRef(null);
  const markerLayerRef = useRef(null);
  const poiLayerRef = useRef(null);
  const circleRef = useRef(null);
  const markersMapRef = useRef(new Map());

  useEffect(() => {
    if (!isResizing) return undefined;
    const handlePointerMove = (event) => {
      const newWidth = event.clientX - 96;
      if (newWidth >= 460 && newWidth <= 960) setSidebarWidth(newWidth);
    };
    const handlePointerUp = () => {
      setIsResizing(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.addEventListener('pointermove', handlePointerMove);
    document.addEventListener('pointerup', handlePointerUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    return () => {
      document.removeEventListener('pointermove', handlePointerMove);
      document.removeEventListener('pointerup', handlePointerUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  useEffect(() => {
    const meta = document.querySelector('meta[name="color-scheme"]');
    const resolvedTheme = theme === 'auto'
      ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : theme;
    if (meta) meta.content = resolvedTheme;
    document.documentElement.setAttribute('content', resolvedTheme);
    localStorage.setItem('color-scheme', theme);
  }, [theme]);

  useEffect(() => {
    async function loadCatalog() {
      try {
        const res = await fetch('/api/listings');
        const payload = await res.json();
        const items = payload.items || [];
        setAllListings(items);
        setProvinces([...new Set(items.map((item) => item.province).filter(Boolean))].sort());
      } catch (err) {
        console.error('Failed to load listings catalog:', err);
      }
    }
    loadCatalog();
  }, []);

  useEffect(() => {
    let source = allListings;
    if (ragFilterOnly && lastChatListings) source = lastChatListings;

    const minPriceVal = parseFloat(minPrice) * 1_000_000_000 || 0;
    const maxPriceVal = parseFloat(maxPrice) * 1_000_000_000 || Infinity;
    const minAreaVal = parseFloat(minArea) || 0;
    const maxAreaVal = parseFloat(maxArea) || Infinity;
    const q = searchQuery.trim().toLowerCase();

    const filtered = source.filter((item) => {
      if (listingType && item.listing_type !== listingType) return false;
      if (provinceSelect && item.province !== provinceSelect) return false;
      if (item.price_vnd && (item.price_vnd < minPriceVal || item.price_vnd > maxPriceVal)) return false;
      if (item.area_m2 && (item.area_m2 < minAreaVal || item.area_m2 > maxAreaVal)) return false;
      if (bedsSelect === '4' && (!item.bedrooms || item.bedrooms < 4)) return false;
      if (bedsSelect && bedsSelect !== '4' && (!item.bedrooms || parseInt(item.bedrooms, 10) !== parseInt(bedsSelect, 10))) return false;
      if (q) {
        const haystack = [item.title, item.address, item.project, item.property_type, item.district].join(' ').toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });

    if (sortBy === 'price_asc') filtered.sort((a, b) => (a.price_vnd || Infinity) - (b.price_vnd || Infinity));
    if (sortBy === 'price_desc') filtered.sort((a, b) => (b.price_vnd || 0) - (a.price_vnd || 0));
    if (sortBy === 'area_desc') filtered.sort((a, b) => (b.area_m2 || 0) - (a.area_m2 || 0));

    setListings(filtered);
  }, [allListings, lastChatListings, searchQuery, listingType, provinceSelect, minPrice, maxPrice, minArea, maxArea, bedsSelect, sortBy, ragFilterOnly]);

  useEffect(() => {
    const L = getLeaflet();
    if (!L || mapRef.current) return;
    const leafletMap = L.map('map', { zoomControl: false }).setView([10.7769, 106.7009], 12);
    L.control.zoom({ position: 'bottomright' }).addTo(leafletMap);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(leafletMap);
    markerLayerRef.current = L.layerGroup().addTo(leafletMap);
    poiLayerRef.current = L.layerGroup().addTo(leafletMap);
    mapRef.current = leafletMap;
  }, []);

  const handleSelectListing = useCallback((id, openPopup = true) => {
    setActiveId(id);
    setActiveTab('details');
    const item = allListings.find((listing) => listing.id === id);
    if (item?.lat && item?.lng && mapRef.current) {
      mapRef.current.panTo([item.lat, item.lng], { animate: true });
      const marker = markersMapRef.current.get(id);
      if (marker && openPopup) marker.openPopup();
    }
  }, [allListings]);

  useEffect(() => {
    const L = getLeaflet();
    if (!L || !mapRef.current || !markerLayerRef.current) return;

    markerLayerRef.current.clearLayers();
    markersMapRef.current.clear();

    listings.forEach((item) => {
      if (!item.lat || !item.lng) return;
      const iconClass = item.id === activeId ? 'pin-marker active' : 'pin-marker';
      const rentClass = item.listing_type === 'cho-thue' ? ' rent' : '';
      const iconChar = item.listing_type === 'cho-thue' ? '<i class="fa-solid fa-key"></i>' : '<i class="fa-solid fa-house"></i>';
      const popupHtml = `
        <div class="map-popup">
          <h3>${escapeHtml(item.title || 'Tin đăng')}</h3>
          <div class="popup-meta">
            <span class="badge-tag ${item.listing_type === 'ban' ? 'sale' : 'rent'}">${escapeHtml(listingModeLabel(item))}</span>
            <span class="badge-tag price">${escapeHtml(formatMoney(item.price_vnd))}</span>
            ${item.area_m2 ? `<span class="badge-tag">${escapeHtml(item.area_m2)} m²</span>` : ''}
          </div>
          <p>${escapeHtml(item.address || 'Chưa có địa chỉ')}</p>
          <small>${escapeHtml(geoPrecisionLabel(item.geo_precision))}</small>
        </div>
      `;

      const customIcon = L.divIcon({
        className: '',
        html: `<div class="${iconClass}${rentClass}">${iconChar}</div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 32],
        popupAnchor: [0, -32],
      });
      const marker = L.marker([item.lat, item.lng], { icon: customIcon })
        .bindPopup(popupHtml)
        .on('click', () => handleSelectListing(item.id, false));
      marker.addTo(markerLayerRef.current);
      markersMapRef.current.set(item.id, marker);
    });
  }, [listings, activeId, handleSelectListing]);

  useEffect(() => {
    const L = getLeaflet();
    if (!L || !mapRef.current) return;
    if (circleRef.current) {
      mapRef.current.removeLayer(circleRef.current);
      circleRef.current = null;
    }
    const activeItem = allListings.find((listing) => listing.id === activeId);
    if (activeItem?.lat && activeItem?.lng) {
      circleRef.current = L.circle([activeItem.lat, activeItem.lng], {
        radius: 1500,
        color: 'hsl(var(--primary-raw))',
        fillColor: 'hsl(var(--primary-raw))',
        fillOpacity: 0.04,
        weight: 1.5,
        dashArray: '4 4',
      }).addTo(mapRef.current);
    }
  }, [activeId, allListings]);

  useEffect(() => {
    const L = getLeaflet();
    if (!activeId) {
      setPois([]);
      if (poiLayerRef.current) poiLayerRef.current.clearLayers();
      return;
    }

    let isMounted = true;
    async function loadPois() {
      setIsPoisLoading(true);
      if (poiLayerRef.current) poiLayerRef.current.clearLayers();
      try {
        const res = await fetch(`/api/listings/${encodeURIComponent(activeId)}/pois`);
        const payload = await res.json();
        const foundPois = payload.pois || [];
        if (!isMounted) return;
        setPois(foundPois);
        if (!L || !poiLayerRef.current) return;
        foundPois.forEach((poi) => {
          if (!poi.lat || !poi.lng) return;
          const iconByCategory = {
            transit_station: 'fa-train-subway',
            school: 'fa-graduation-cap',
            hospital: 'fa-house-medical',
            park: 'fa-tree',
          };
          const customIcon = L.divIcon({
            className: '',
            html: `<div class="poi-marker ${escapeHtml(poi.category)}"><i class="fa-solid ${iconByCategory[poi.category] || 'fa-location-dot'}"></i></div>`,
            iconSize: [22, 22],
            iconAnchor: [11, 11],
          });
          L.marker([poi.lat, poi.lng], { icon: customIcon })
            .bindPopup(`<strong>${escapeHtml(poi.name)}</strong><br><small>${escapeHtml(poi.address || '')}</small>`)
            .addTo(poiLayerRef.current);
        });
      } catch (err) {
        console.error('Failed to load POIs:', err);
        if (isMounted) setPois([]);
      } finally {
        if (isMounted) setIsPoisLoading(false);
      }
    }
    loadPois();
    return () => {
      isMounted = false;
    };
  }, [activeId]);

  const handleFitMap = useCallback(() => {
    const L = getLeaflet();
    const activeMarkers = [...markersMapRef.current.values()];
    if (!L || !activeMarkers.length || !mapRef.current) return;
    const bounds = L.latLngBounds(activeMarkers.map((marker) => marker.getLatLng()));
    mapRef.current.fitBounds(bounds.pad(0.15), { maxZoom: 14 });
  }, []);

  useEffect(() => {
    if (!pendingFitMap) return;
    const timer = window.setTimeout(() => {
      handleFitMap();
      setPendingFitMap(false);
    }, 80);
    return () => window.clearTimeout(timer);
  }, [pendingFitMap, listings, handleFitMap]);

  const activeListing = useMemo(() => allListings.find((listing) => listing.id === activeId), [allListings, activeId]);

  const loanPrincipal = (activeListing?.price_vnd || 0) * (1 - downPaymentPct / 100);
  const monthlyInterestRate = (Number(interestRate) / 100) / 12;
  const loanTermMonths = Number(loanTermYears) * 12;
  const monthlyMortgagePayment = loanPrincipal > 0 && loanTermMonths > 0
    ? monthlyInterestRate > 0
      ? loanPrincipal * (monthlyInterestRate * (1 + monthlyInterestRate) ** loanTermMonths) / (((1 + monthlyInterestRate) ** loanTermMonths) - 1)
      : loanPrincipal / loanTermMonths
    : 0;
  const totalInterestPayable = (monthlyMortgagePayment * loanTermMonths) - loanPrincipal;
  const incomeRatio = Number(monthlyIncome) > 0 && monthlyMortgagePayment > 0
    ? (monthlyMortgagePayment / (Number(monthlyIncome) * 1_000_000)) * 100
    : null;
  const affordabilityLevel = incomeRatio === null ? 'neutral' : incomeRatio <= 40 ? 'success' : incomeRatio <= 60 ? 'warning' : 'danger';

  const setPrompt = (prompt) => {
    setChatInput(prompt);
    setActiveTab('chat');
  };

  const toggleCompare = (item) => {
    setCompareList((current) => {
      if (current.some((candidate) => candidate.id === item.id)) return current.filter((candidate) => candidate.id !== item.id);
      if (current.length >= 2) return [current[1], item];
      return [...current, item];
    });
  };

  const buildComparePrompt = (items = compareList) => {
    if (items.length !== 2) return '';
    return `Hãy so sánh chi tiết 2 bất động sản sau và đưa ra khuyến nghị dựa trên giá, diện tích, pháp lý, vị trí, tiện ích xung quanh, cảm quan cư dân và khả năng tài chính:\n\nBĐS 1: ${items[0].title}\n- Giá: ${formatMoney(items[0].price_vnd)}\n- Diện tích: ${items[0].area_m2 || 'N/A'} m²\n- Đơn giá: ${formatPricePerM2(items[0])}\n- Vị trí: ${items[0].address || 'N/A'}\n- Pháp lý: ${items[0].legal || 'Đang cập nhật'}\n\nBĐS 2: ${items[1].title}\n- Giá: ${formatMoney(items[1].price_vnd)}\n- Diện tích: ${items[1].area_m2 || 'N/A'} m²\n- Đơn giá: ${formatPricePerM2(items[1])}\n- Vị trí: ${items[1].address || 'N/A'}\n- Pháp lý: ${items[1].legal || 'Đang cập nhật'}`;
  };

  const handleChatSubmit = async (event, overrideMessage) => {
    if (event) event.preventDefault();
    const cleanMsg = (overrideMessage ?? chatInput).trim();
    if (!cleanMsg || isChatLoading) return;

    setChatMessages((prev) => [
      ...prev,
      { role: 'user', text: cleanMsg },
      { role: 'assistant', text: '', streaming: true, thinkingProcess: [], statusText: null, payload: null },
    ]);
    setChatInput('');
    setIsChatLoading(true);
    setActiveTab('chat');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: cleanMsg }),
      });
      if (!res.ok || !res.body) throw new Error(`Server error: ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamDone = false;

      while (!streamDone) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split('\n\n');
        buffer = messages.pop() || '';

        for (const rawMsg of messages) {
          if (!rawMsg.trim()) continue;
          const lines = rawMsg.split('\n');
          const eventType = lines.find((line) => line.startsWith('event: '))?.slice(7).trim();
          const dataStr = lines.find((line) => line.startsWith('data: '))?.slice(6).trim();
          if (!eventType || !dataStr) continue;

          if (eventType === 'metadata') {
            const meta = JSON.parse(dataStr);
            setChatMessages((prev) => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
                updated[lastIdx] = { ...updated[lastIdx], payload: meta, isError: Boolean(meta.error) };
              }
              return updated;
            });
            setRagStatus({ label: meta.error ? 'Fallback' : 'RAG', isError: Boolean(meta.error) });
            if (meta.listings?.length) {
              const ids = new Set(meta.listings.map((listing) => listing.id));
              setLastChatListings(allListings.filter((item) => ids.has(item.id)));
            } else {
              setLastChatListings(null);
            }
          } else if (eventType === 'thought' || eventType === 'observation') {
            const text = JSON.parse(dataStr);
            setChatMessages((prev) => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
                updated[lastIdx] = {
                  ...updated[lastIdx],
                  thinkingProcess: [...(updated[lastIdx].thinkingProcess || []), { type: eventType, text }],
                };
              }
              return updated;
            });
          } else if (eventType === 'status') {
            const statusMsg = JSON.parse(dataStr);
            setChatMessages((prev) => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') updated[lastIdx] = { ...updated[lastIdx], statusText: statusMsg };
              return updated;
            });
          } else if (eventType === 'chunk') {
            const token = JSON.parse(dataStr);
            setChatMessages((prev) => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
                updated[lastIdx] = { ...updated[lastIdx], text: updated[lastIdx].text + token, statusText: null };
              }
              return updated;
            });
          } else if (eventType === 'done') {
            setChatMessages((prev) => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') updated[lastIdx] = { ...updated[lastIdx], streaming: false, statusText: null };
              return updated;
            });
            streamDone = true;
            break;
          }
        }
      }
    } catch (err) {
      console.error('Chat submit failed:', err);
      setChatMessages((prev) => [...prev, {
        role: 'assistant',
        text: `Lỗi kết nối RAG server: ${err.message}`,
        isError: true,
      }]);
      setRagStatus({ label: 'Offline', isError: true });
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleFocusRAGListings = () => {
    setActiveTab('search');
    setRagFilterOnly(true);
    setPendingFitMap(true);
  };

  const handleSubmitCompare = () => {
    const prompt = buildComparePrompt();
    setShowCompareModal(false);
    handleChatSubmit(null, prompt);
  };

  return (
    <div className="app-shell" style={{ '--panel-width': `${sidebarWidth}px` }}>
      <nav className="sidebar-nav" aria-label="Menu chức năng">
        <div className="nav-top">
          {[
            ['chat', 'fa-robot', 'Chat'],
            ['search', 'fa-magnifying-glass', 'Tìm kiếm'],
            ['details', 'fa-circle-info', 'Chi tiết'],
          ].map(([tab, icon, label]) => (
            <button key={tab} type="button" className={`nav-item ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)} title={label}>
              <i className={`fa-solid ${icon}`} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <div className="nav-bottom">
          <button type="button" className="nav-item" onClick={() => setTheme((prev) => prev === 'dark' ? 'light' : prev === 'light' ? 'auto' : 'dark')} title={`Giao diện: ${theme}`}>
            <i className="fa-solid fa-circle-half-stroke" />
            <span>Giao diện</span>
          </button>
        </div>
      </nav>

      <aside className="sidebar-panel">
        {activeTab === 'chat' && (
          <section id="chatTab" className="tab-content active" aria-label="Trợ lý chat">
            <header className="panel-header rag-first-header">
              <div>
                <h1>Trợ lý BĐS RAG</h1>
                <p>Hiểu nhu cầu tự nhiên, truy xuất nguồn tin, POIs, cộng đồng và tài chính.</p>
              </div>
              <span className={`status-pill ${ragStatus.isError ? 'error' : ''}`}>{ragStatus.label}</span>
            </header>

            <div className="prompt-group-grid" aria-label="Gợi ý câu hỏi theo năng lực">
              {promptGroups.map((group) => (
                <button key={group.title} type="button" className="prompt-group-card" onClick={() => setPrompt(group.prompt)}>
                  <i className={`fa-solid ${group.icon}`} />
                  <span>{group.title}</span>
                </button>
              ))}
            </div>

            <div className="chat-messages-container" aria-live="polite">
              {chatMessages.map((msg, index) => (
                <article key={`${msg.role}-${index}`} className={`message ${msg.role} ${msg.isError ? 'error' : ''} animate-fade-in`}>
                  <div className="message-header">
                    <div className="avatar">
                      <i className={`fa-solid ${msg.role === 'user' ? 'fa-user' : 'fa-robot'}`} />
                    </div>
                    <span className="sender-label">{msg.role === 'user' ? 'Khách hàng' : 'Trợ lý AI'}</span>
                    {msg.streaming && (
                      <span className="streaming-badge"><i className="fa-solid fa-circle-notch fa-spin" /> {msg.statusText || 'Đang tạo...'}</span>
                    )}
                  </div>
                  <div className="message-body">
                    {msg.role === 'user' ? (
                      <p>{msg.text}</p>
                    ) : (
                      <>
                        {msg.thinkingProcess?.length > 0 && (
                          <details className="thinking-process-panel">
                            <summary><i className="fa-solid fa-brain" /> Luồng ReAct</summary>
                            <div className="thinking-content">
                              {msg.thinkingProcess.map((step, idx) => (
                                <div key={`${step.type}-${idx}`} className={`think-step ${step.type}`}>
                                  <strong>{step.type === 'thought' ? 'Agent' : 'System'}:</strong>
                                  <pre>{step.text}</pre>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                        {msg.text ? (
                          <div>
                            <div dangerouslySetInnerHTML={renderSafeMarkdown(msg.text)} />
                            {msg.streaming && <span className="stream-cursor">▌</span>}
                          </div>
                        ) : msg.streaming ? (
                          <div className="typing-dots"><span /><span /><span /></div>
                        ) : null}
                      </>
                    )}

                    {msg.payload && (
                      <>
                        <div className="chat-meta-tag" title={`Intent: ${msg.payload.intent}`}>
                          <i className={`fa-solid ${msg.payload.llm_used ? 'fa-microchip' : 'fa-network-wired'}`} />
                          {msg.payload.intent} • {Object.keys(msg.payload.filters_applied || {}).length} filters • {(msg.payload.sources || []).length} sources
                        </div>

                        {msg.payload.listings?.length > 0 && (
                          <div className="recommendations-panel">
                            <div className="recommendations-heading">
                              <span><i className="fa-solid fa-house-circle-check" /> BĐS được truy xuất</span>
                              <button type="button" className="btn-soft" onClick={handleFocusRAGListings}>
                                <i className="fa-solid fa-map-location-dot" /> Ghim {msg.payload.listings.length} tin
                              </button>
                            </div>
                            {msg.payload.listings.slice(0, 4).map((item) => (
                              <RecommendationCard
                                key={item.id}
                                item={item}
                                onSelect={handleSelectListing}
                                onCompare={toggleCompare}
                                selectedForCompare={compareList.some((candidate) => candidate.id === item.id)}
                              />
                            ))}
                          </div>
                        )}

                        {msg.payload.sources?.length > 0 && (
                          <>
                            <div className="sources-carousel-title">
                              <i className="fa-solid fa-book-open" /> Tài liệu tham khảo RAG
                            </div>
                            <div className="sources-carousel">
                              {msg.payload.sources.map((source, sIdx) => {
                                const badgeLabel = source.collection === 'social_neighborhood'
                                  ? 'Ý kiến MXH'
                                  : source.collection === 'projects'
                                    ? 'Dự án'
                                    : source.collection === 'articles'
                                      ? 'Tin tức'
                                      : 'Bài viết';
                                return (
                                  <div key={`${source.collection}-${sIdx}`} className="source-card animate-fade-in" onClick={() => source.url && source.url !== 'None' && window.open(source.url, '_blank', 'noopener,noreferrer')} title={`Độ khớp: ${((source.score || 0) * 100).toFixed(1)}%`}>
                                    <span className="source-type-badge">{badgeLabel}</span>
                                    <p>{source.text}</p>
                                  </div>
                                );
                              })}
                            </div>
                          </>
                        )}
                      </>
                    )}
                  </div>
                </article>
              ))}
            </div>

            <form className="chat-input-form" onSubmit={handleChatSubmit}>
              <div className="input-wrapper">
                <textarea
                  id="chatInput"
                  rows={1}
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="Nhập nhu cầu hoặc câu hỏi BĐS bằng tiếng Việt..."
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault();
                      handleChatSubmit();
                    }
                  }}
                />
                <button type="submit" disabled={isChatLoading} aria-label="Gửi câu hỏi">
                  <i className={`fa-solid ${isChatLoading ? 'fa-circle-notch fa-spin' : 'fa-paper-plane'}`} />
                </button>
              </div>
            </form>
          </section>
        )}

        {activeTab === 'search' && (
          <section id="searchTab" className="tab-content active" aria-label="Tìm kiếm nâng cao">
            <header className="panel-header">
              <div>
                <h2>Tìm kiếm & Bộ lọc</h2>
                <p>Hiển thị {listings.length} bất động sản khớp với bối cảnh hiện tại.</p>
              </div>
              <button type="button" className="btn-icon" onClick={handleFitMap} title="Căn chỉnh bản đồ">
                <i className="fa-solid fa-expand" />
              </button>
            </header>

            <div className="filter-controls-container">
              <div className="filter-group">
                <label htmlFor="searchInput">Từ khóa</label>
                <div className="input-icon-wrapper">
                  <i className="fa-solid fa-magnifying-glass" />
                  <input id="searchInput" type="search" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Tên dự án, đường, quận..." />
                </div>
              </div>
              <div className="filter-row-grid">
                <div className="filter-group">
                  <label htmlFor="listingType">Giao dịch</label>
                  <select id="listingType" value={listingType} onChange={(event) => setListingType(event.target.value)}>
                    <option value="">Tất cả</option>
                    <option value="ban">Mua bán</option>
                    <option value="cho-thue">Cho thuê</option>
                  </select>
                </div>
                <div className="filter-group">
                  <label htmlFor="provinceSelect">Tỉnh / Thành</label>
                  <select id="provinceSelect" value={provinceSelect} onChange={(event) => setProvinceSelect(event.target.value)}>
                    <option value="">Tất cả</option>
                    {provinces.map((province) => <option key={province} value={province}>{province}</option>)}
                  </select>
                </div>
              </div>
              <div className="filter-group">
                <div className="filter-label-row"><span>Khoảng giá</span><span className="range-value-display">{minPrice || maxPrice ? `${minPrice || 0} - ${maxPrice || '∞'} Tỷ` : 'Tất cả'}</span></div>
                <div className="range-inputs">
                  <input type="number" placeholder="Min (Tỷ)" min="0" step="0.1" value={minPrice} onChange={(event) => setMinPrice(event.target.value)} />
                  <input type="number" placeholder="Max (Tỷ)" min="0" step="0.1" value={maxPrice} onChange={(event) => setMaxPrice(event.target.value)} />
                </div>
              </div>
              <div className="filter-group">
                <div className="filter-label-row"><span>Diện tích (m²)</span><span className="range-value-display">{minArea || maxArea ? `${minArea || 0} - ${maxArea || '∞'} m²` : 'Tất cả'}</span></div>
                <div className="range-inputs">
                  <input type="number" placeholder="Min" min="0" value={minArea} onChange={(event) => setMinArea(event.target.value)} />
                  <input type="number" placeholder="Max" min="0" value={maxArea} onChange={(event) => setMaxArea(event.target.value)} />
                </div>
              </div>
              <div className="filter-row-grid">
                <div className="filter-group">
                  <label htmlFor="bedsSelect">Phòng ngủ</label>
                  <select id="bedsSelect" value={bedsSelect} onChange={(event) => setBedsSelect(event.target.value)}>
                    <option value="">Tất cả</option>
                    <option value="1">1 PN</option>
                    <option value="2">2 PN</option>
                    <option value="3">3 PN</option>
                    <option value="4">4+ PN</option>
                  </select>
                </div>
                <div className="filter-group">
                  <label htmlFor="sortBySelect">Sắp xếp</label>
                  <select id="sortBySelect" value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
                    <option value="">Mặc định</option>
                    <option value="price_asc">Giá: Thấp đến Cao</option>
                    <option value="price_desc">Giá: Cao đến Thấp</option>
                    <option value="area_desc">Diện tích: Lớn đến Nhỏ</option>
                  </select>
                </div>
              </div>
              {lastChatListings && (
                <div className="rag-toggle-container">
                  <label className="switch-label">
                    <input type="checkbox" checked={ragFilterOnly} onChange={(event) => setRagFilterOnly(event.target.checked)} />
                    <span className="slider-switch" />
                    <span>Chỉ hiện tin từ kết quả RAG gần nhất</span>
                  </label>
                </div>
              )}
            </div>

            <div className="stats-row" aria-label="Thống kê">
              <div className="stat-card"><span className="stat-val">{listings.length}</span><span className="stat-lbl">Tổng tin</span></div>
              <div className="stat-card"><span className="stat-val text-sale">{listings.filter((item) => item.listing_type === 'ban').length}</span><span className="stat-lbl">Tin bán</span></div>
              <div className="stat-card"><span className="stat-val text-rent">{listings.filter((item) => item.listing_type === 'cho-thue').length}</span><span className="stat-lbl">Tin thuê</span></div>
            </div>

            <div id="listingList" className="listing-list-scroll">
              {listings.length === 0 ? (
                <div className="poi-empty">
                  <i className="fa-solid fa-house-circle-xmark" /> Không có tin đăng nào khớp với bộ lọc của bạn.
                </div>
              ) : (
                listings.map((item) => (
                  <ListingCard
                    key={item.id}
                    item={item}
                    active={item.id === activeId}
                    onSelect={handleSelectListing}
                    onCompare={toggleCompare}
                    selectedForCompare={compareList.some((candidate) => candidate.id === item.id)}
                  />
                ))
              )}
            </div>
          </section>
        )}

        {activeTab === 'details' && (
          <section id="detailsTab" className="tab-content active" aria-label="Chi tiết bất động sản">
            {!activeId || !activeListing ? (
              <div className="details-placeholder animate-fade-in">
                <i className="fa-solid fa-house-circle-exclamation" />
                <h3>Chưa chọn bất động sản</h3>
                <p>Chọn một tin đăng hoặc ghim trên bản đồ để xem quyết định chi tiết.</p>
              </div>
            ) : (
              <div className="details-content-scroll decision-dashboard">
                <div className="detail-image-wrapper">
                  {activeListing.image ? <img src={activeListing.image} alt={activeListing.title || 'Bất động sản'} /> : <div className="detail-image-placeholder"><i className="fa-solid fa-image" /></div>}
                </div>
                <div className="detail-heading-block">
                  <span className={`badge-tag ${activeListing.listing_type === 'ban' ? 'sale' : 'rent'}`}>{listingModeLabel(activeListing)}</span>
                  <h3 className="detail-title">{activeListing.title || 'Tin đăng chưa có tiêu đề'}</h3>
                  <p className="detail-location-text"><i className="fa-solid fa-location-dot" /> {activeListing.address || 'Chưa có địa chỉ'}</p>
                </div>

                <div className="decision-summary-grid">
                  <div><strong>{formatMoney(activeListing.price_vnd)}</strong><span>Giá niêm yết</span></div>
                  <div><strong>{activeListing.area_m2 ? `${activeListing.area_m2} m²` : 'Thiếu'}</strong><span>Diện tích</span></div>
                  <div><strong>{formatPricePerM2(activeListing)}</strong><span>Đơn giá</span></div>
                  <div><strong>{geoPrecisionLabel(activeListing.geo_precision)}</strong><span>Độ tin cậy bản đồ</span></div>
                </div>

                <div className="evidence-strip">
                  <span><i className="fa-solid fa-file-shield" /> {activeListing.legal || 'Pháp lý đang cập nhật'}</span>
                  <span><i className="fa-solid fa-couch" /> {activeListing.furniture || 'Nội thất đang cập nhật'}</span>
                  <span><i className="fa-solid fa-bed" /> {activeListing.bedrooms ? `${activeListing.bedrooms} PN` : 'PN: N/A'}</span>
                  <span><i className="fa-solid fa-bath" /> {activeListing.bathrooms ? `${activeListing.bathrooms} WC` : 'WC: N/A'}</span>
                </div>

                <section className="decision-section">
                  <h4><i className="fa-solid fa-location-dot" /> Tiện ích xung quanh</h4>
                  {isPoisLoading ? (
                    <p className="muted-line">Đang tải POIs quanh bất động sản...</p>
                  ) : pois.length ? (
                    <div className="pois-grid">
                      {pois.slice(0, 8).map((poi, index) => (
                        <div key={`${poi.name}-${index}`} className={`poi-item ${poi.category || ''}`}>
                          <span className="poi-name">{poi.name || 'POI chưa đặt tên'}</span>
                          <small>{poiLabels[poi.category] || poi.category || 'Tiện ích'}</small>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="muted-line">Chưa có POIs liên kết trong cơ sở dữ liệu cho tin này.</p>
                  )}
                </section>

                <section className="decision-section mortgage-panel">
                  <h4><i className="fa-solid fa-calculator" /> Hoạch định tài chính</h4>
                  <div className="mortgage-input-grid">
                    <label>Trả trước (%)<input type="number" min="0" max="100" value={downPaymentPct} onChange={(event) => setDownPaymentPct(event.target.value)} /></label>
                    <label>Lãi suất/năm (%)<input type="number" min="0" step="0.1" value={interestRate} onChange={(event) => setInterestRate(event.target.value)} /></label>
                    <label>Thời hạn (năm)<input type="number" min="1" value={loanTermYears} onChange={(event) => setLoanTermYears(event.target.value)} /></label>
                    <label>Thu nhập/tháng (triệu)<input type="number" min="0" value={monthlyIncome} onChange={(event) => setMonthlyIncome(event.target.value)} placeholder="VD: 60" /></label>
                  </div>
                  <div className={`affordability-card ${affordabilityLevel}`}>
                    <strong>{formatMonthlyMoney(monthlyMortgagePayment)}</strong>
                    <span>Ước tính trả góp hàng tháng</span>
                    <small>Tổng lãi dự kiến: {formatMoney(totalInterestPayable > 0 ? totalInterestPayable : 0)}{incomeRatio !== null ? ` • Chiếm ${incomeRatio.toFixed(0)}% thu nhập` : ''}</small>
                  </div>
                </section>

                <div className="detail-action-row">
                  {activeListing.url && (
                    <a href={activeListing.url} target="_blank" rel="noreferrer" className="btn-api">
                      <i className="fa-solid fa-arrow-up-right-from-square" /> Tin gốc
                    </a>
                  )}
                  <button type="button" className="btn-soft" onClick={() => toggleCompare(activeListing)}>
                    <i className="fa-solid fa-scale-balanced" /> {compareList.some((item) => item.id === activeListing.id) ? 'Bỏ so sánh' : 'Thêm so sánh'}
                  </button>
                </div>
              </div>
            )}
          </section>
        )}

        {compareList.length > 0 && !showCompareModal && (
          <div className="compare-floating-bar card-glass animate-slide-up">
            <span>Đã chọn {compareList.length}/2 BĐS</span>
            <div>
              <button type="button" className="btn-api" disabled={compareList.length !== 2} onClick={() => setShowCompareModal(true)}>
                So sánh
              </button>
              <button type="button" className="btn-soft" onClick={() => setCompareList([])} aria-label="Xóa danh sách so sánh">
                <i className="fa-solid fa-times" />
              </button>
            </div>
          </div>
        )}

        <div
          className={`sidebar-resizer ${isResizing ? 'is-resizing' : ''}`}
          role="separator"
          aria-label="Kéo để thay đổi chiều rộng khung chat"
          aria-orientation="vertical"
          onPointerDown={(event) => {
            event.preventDefault();
            event.currentTarget.setPointerCapture(event.pointerId);
            setIsResizing(true);
          }}
          title="Kéo ngang để đổi chiều rộng khung chat"
        >
          <span className="sidebar-resizer-grip" />
        </div>
      </aside>

      {showCompareModal && compareList.length === 2 && (
        <CompareTable
          items={compareList}
          onClose={() => setShowCompareModal(false)}
          onSubmitAnalysis={handleSubmitCompare}
        />
      )}

      <main className="map-area">
        <div className="map-overlay-toolbar card-glass animate-slide-down">
          <div className="toolbar-left">
            <span className="app-logo"><i className="fa-solid fa-map-location-dot" /> Bản đồ ngữ cảnh</span>
            <span id="mapTitle">{listings.length} ghim theo nhu cầu</span>
          </div>
          <div className="toolbar-right">
            <span id="geoNote" className="geo-badge"><i className="fa-solid fa-location-crosshairs" /> Có vị trí ước lượng</span>
            <a className="btn-api" href="/docs" target="_blank" rel="noreferrer" title="Swagger API Documentation"><i className="fa-solid fa-code" /> API Docs</a>
          </div>
        </div>
        <div id="map" />
      </main>
    </div>
  );
}
