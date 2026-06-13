import React, { useState, useEffect, useRef } from 'react';
import { marked } from 'marked';

// --- Money/VND Formatter ---
const formatMoney = (vnd) => {
  if (!vnd) return "Thỏa thuận";
  if (vnd >= 1_000_000_000) {
    return `${(vnd / 1_000_000_000).toFixed(1).replace(".0", "")} tỷ`;
  }
  return `${Math.round(vnd / 1_000_000)} triệu`;
};

export default function App() {
  // --- Core Application States ---
  const [activeTab, setActiveTab] = useState('chat');
  const [theme, setTheme] = useState(localStorage.getItem('color-scheme') || 'auto');
  
  const [allListings, setAllListings] = useState([]);
  const [listings, setListings] = useState([]);
  const [lastChatListings, setLastChatListings] = useState(null);
  const [activeId, setActiveId] = useState(null);

  // Chat State
  const [chatMessages, setChatMessages] = useState([
    {
      role: 'assistant',
      text: 'Chào bạn! Hãy đặt câu hỏi về nhu cầu mua/thuê nhà, ngân sách, vị trí mong muốn hoặc yêu cầu phân tích tài chính vay mua nhà.\n\nTôi sẽ phân tích dựa trên dữ liệu RAG thực tế, ghim các vị trí tương ứng trên bản đồ và thống kê chi tiết tiện ích xung quanh.',
      payload: null
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [ragStatus, setRagStatus] = useState({ label: 'RAG', isError: false });

  // Filters State
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

  // Provinces List
  const [provinces, setProvinces] = useState([]);

  // Active Listing Detail & POIs State
  const [pois, setPois] = useState([]);
  const [isPoisLoading, setIsPoisLoading] = useState(false);

  // Mortgage Calculator State
  const [downPaymentPct, setDownPaymentPct] = useState(30);
  const [interestRate, setInterestRate] = useState(9.0);
  const [loanTermYears, setLoanTermYears] = useState(20);
  const [monthlyIncome, setMonthlyIncome] = useState('');

  // Map Refs
  const mapRef = useRef(null);
  const markerLayerRef = useRef(null);
  const poiLayerRef = useRef(null);
  const circleRef = useRef(null);
  const markersMapRef = useRef(new Map());

  // --- Initialize theme ---
  useEffect(() => {
    const meta = document.querySelector('meta[name="color-scheme"]');
    if (theme === 'auto') {
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      meta.content = isDark ? 'dark' : 'light';
      document.documentElement.setAttribute('content', isDark ? 'dark' : 'light');
    } else {
      meta.content = theme;
      document.documentElement.setAttribute('content', theme);
    }
    localStorage.setItem('color-scheme', theme);
  }, [theme]);

  // --- Fetch initial catalog listings ---
  useEffect(() => {
    async function loadCatalog() {
      try {
        const res = await fetch('/api/listings');
        const payload = await res.json();
        setAllListings(payload.items);
        
        // Extract unique provinces
        const uniqueProvinces = [...new Set(payload.items.map(i => i.province).filter(Boolean))].sort();
        setProvinces(uniqueProvinces);
      } catch (err) {
        console.error('Failed to load listings catalog:', err);
      }
    }
    loadCatalog();
  }, []);

  // --- Apply dynamic filters whenever search parameters update ---
  useEffect(() => {
    let source = allListings;
    if (ragFilterOnly && lastChatListings) {
      source = lastChatListings;
    }

    const minPriceVal = parseFloat(minPrice) * 1_000_000_000 || 0;
    const maxPriceVal = parseFloat(maxPrice) * 1_000_000_000 || Infinity;
    const minAreaVal = parseFloat(minArea) || 0;
    const maxAreaVal = parseFloat(maxArea) || Infinity;
    const q = searchQuery.trim().toLowerCase();

    let filtered = source.filter(item => {
      if (listingType && item.listing_type !== listingType) return false;
      if (provinceSelect && item.province !== provinceSelect) return false;

      const price = item.price_vnd || 0;
      if (price && (price < minPriceVal || price > maxPriceVal)) return false;

      const area = item.area_m2 || 0;
      if (area && (area < minAreaVal || area > maxAreaVal)) return false;

      if (bedsSelect) {
        if (bedsSelect === '4') {
          if (!item.bedrooms || item.bedrooms < 4) return false;
        } else {
          if (!item.bedrooms || parseInt(item.bedrooms) !== parseInt(bedsSelect)) return false;
        }
      }

      if (q) {
        const haystack = [item.title, item.address, item.project, item.property_type, item.district]
          .join(" ").toLowerCase();
        if (!haystack.includes(q)) return false;
      }

      return true;
    });

    // Sort listings
    if (sortBy === 'price_asc') {
      filtered.sort((a, b) => (a.price_vnd || Infinity) - (b.price_vnd || Infinity));
    } else if (sortBy === 'price_desc') {
      filtered.sort((a, b) => (b.price_vnd || 0) - (a.price_vnd || 0));
    } else if (sortBy === 'area_desc') {
      filtered.sort((a, b) => (b.area_m2 || 0) - (a.area_m2 || 0));
    }

    setListings(filtered);
  }, [allListings, lastChatListings, searchQuery, listingType, provinceSelect, minPrice, maxPrice, minArea, maxArea, bedsSelect, sortBy, ragFilterOnly]);

  // --- Leaflet Map Init effect ---
  useEffect(() => {
    if (!mapRef.current) {
      const leafletMap = L.map('map', { zoomControl: false }).setView([10.7769, 106.7009], 12);
      
      L.control.zoom({ position: 'bottomright' }).addTo(leafletMap);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      }).addTo(leafletMap);

      markerLayerRef.current = L.layerGroup().addTo(leafletMap);
      poiLayerRef.current = L.layerGroup().addTo(leafletMap);
      mapRef.current = leafletMap;
    }
  }, []);

  // --- Render markers on map when listings or active selection changes ---
  useEffect(() => {
    if (!mapRef.current) return;
    
    markerLayerRef.current.clearLayers();
    markersMapRef.current.clear();

    listings.forEach(item => {
      if (!item.lat || !item.lng) return;

      const isActive = item.id === activeId;
      
      // Setup divIcon wrapper
      const iconClass = isActive ? "pin-marker active" : "pin-marker";
      const rentClass = item.listing_type === 'cho-thue' ? " rent" : "";
      const iconChar = item.listing_type === 'cho-thue' ? '<i class="fa-solid fa-key"></i>' : '<i class="fa-solid fa-house"></i>';
      
      const customIcon = L.divIcon({
        className: '',
        html: `<div class="${iconClass}${rentClass}">${iconChar}</div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 32],
        popupAnchor: [0, -32]
      });

      const popupHtml = `
        <div style="min-width: 200px; padding: 4px 0;">
          <h3 style="margin: 0 0 6px; font-size: 13px; font-weight: 700; line-height: 1.4;">${item.title}</h3>
          <div class="popup-meta" style="margin-bottom: 6px;">
            <span class="badge-tag ${item.listing_type === 'ban' ? 'sale' : 'rent'}">
              ${item.listing_type === 'ban' ? 'Bán' : 'Thuê'}
            </span>
            <span class="badge-tag price">${formatMoney(item.price_vnd)}</span>
            ${item.area_m2 ? `<span class="badge-tag">${item.area_m2} m²</span>` : ''}
          </div>
          <p style="margin: 0 0 8px; font-size: 11px; color: var(--text-muted-raw);">${item.address}</p>
        </div>
      `;

      const marker = L.marker([item.lat, item.lng], { icon: customIcon })
        .bindPopup(popupHtml)
        .on('click', () => handleSelectListing(item.id, false));

      marker.addTo(markerLayerRef.current);
      markersMapRef.current.set(item.id, marker);
    });
  }, [listings, activeId]);

  // --- Selection circle effect ---
  useEffect(() => {
    if (!mapRef.current) return;
    
    if (circleRef.current) {
      mapRef.current.removeLayer(circleRef.current);
      circleRef.current = null;
    }

    const activeItem = allListings.find(l => l.id === activeId);
    if (activeItem && activeItem.lat && activeItem.lng) {
      const circle = L.circle([activeItem.lat, activeItem.lng], {
        radius: 1500,
        color: 'hsl(var(--primary-raw))',
        fillColor: 'hsl(var(--primary-raw))',
        fillOpacity: 0.04,
        weight: 1.5,
        dashArray: '4 4'
      }).addTo(mapRef.current);

      circleRef.current = circle;
    }
  }, [activeId, allListings]);

  // --- Fetch and plot POIs when active listing changes ---
  useEffect(() => {
    if (!activeId) {
      setPois([]);
      if (poiLayerRef.current) poiLayerRef.current.clearLayers();
      return;
    }

    async function loadPois() {
      setIsPoisLoading(true);
      if (poiLayerRef.current) poiLayerRef.current.clearLayers();

      try {
        const res = await fetch(`/api/listings/${encodeURIComponent(activeId)}/pois`);
        const payload = await res.json();
        const foundPois = payload.pois || [];
        setPois(foundPois);

        // Plot POIs on map
        foundPois.forEach(poi => {
          if (!poi.lat || !poi.lng) return;

          let iconHtml = '<i class="fa-solid fa-location-dot"></i>';
          if (poi.category === 'transit_station') iconHtml = '<i class="fa-solid fa-train-subway"></i>';
          else if (poi.category === 'school') iconHtml = '<i class="fa-solid fa-graduation-cap"></i>';
          else if (poi.category === 'hospital') iconHtml = '<i class="fa-solid fa-house-medical"></i>';
          else if (poi.category === 'park') iconHtml = '<i class="fa-solid fa-tree"></i>';

          const customIcon = L.divIcon({
            className: '',
            html: `<div class="poi-marker ${poi.category}">${iconHtml}</div>`,
            iconSize: [22, 22],
            iconAnchor: [11, 11]
          });

          L.marker([poi.lat, poi.lng], { icon: customIcon })
            .bindPopup(`<strong>${poi.name}</strong><br><small style="color:var(--text-muted-raw);">${poi.address || ''}</small>`)
            .addTo(poiLayerRef.current);
        });

      } catch (err) {
        console.error('Failed to load POIs:', err);
      } finally {
        setIsPoisLoading(false);
      }
    }
    loadPois();
  }, [activeId]);

  // --- Fit map bounds ---
  const handleFitMap = () => {
    const activeMarkers = [...markersMapRef.current.values()];
    if (!activeMarkers.length || !mapRef.current) return;
    const bounds = L.latLngBounds(activeMarkers.map(m => m.getLatLng()));
    mapRef.current.fitBounds(bounds.pad(0.15), { maxZoom: 14 });
  };

  // --- Select a listing card / pin ---
  const handleSelectListing = (id, openPopup = true) => {
    setActiveId(id);
    setActiveTab('details');

    const item = allListings.find(l => l.id === id);
    if (item && mapRef.current) {
      mapRef.current.panTo([item.lat, item.lng], { animate: true });
      const marker = markersMapRef.current.get(id);
      if (marker && openPopup) {
        marker.openPopup();
      }
    }
  };

  // --- Submit user chat queries ---
  const handleChatSubmit = async (e) => {
    if (e) e.preventDefault();
    const cleanMsg = chatInput.trim();
    if (!cleanMsg || isChatLoading) return;

    setChatMessages(prev => [...prev, { role: 'user', text: cleanMsg }]);
    setChatInput('');
    setIsChatLoading(true);

    // Placeholder assistant message index — will be updated as chunks arrive
    let assistantMsgIndex = null;

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: cleanMsg })
      });

      if (!res.ok || !res.body) {
        throw new Error(`Server error: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (separated by double newline)
        const messages = buffer.split('\n\n');
        buffer = messages.pop(); // Keep the last incomplete part in the buffer

        for (const rawMsg of messages) {
          if (!rawMsg.trim()) continue;

          // Parse SSE fields
          const lines = rawMsg.split('\n');
          let eventType = null;
          let dataStr = null;

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice('event: '.length).trim();
            } else if (line.startsWith('data: ')) {
              dataStr = line.slice('data: '.length).trim();
            }
          }

          if (!eventType || !dataStr) continue;

          if (eventType === 'metadata') {
            const meta = JSON.parse(dataStr);
            // Insert placeholder assistant message
            setChatMessages(prev => {
              assistantMsgIndex = prev.length;
              return [...prev, {
                role: 'assistant',
                text: '',
                payload: meta,
                isError: Boolean(meta.error),
                streaming: true,
              }];
            });
            setRagStatus({ label: meta.error ? 'Fallback' : 'RAG', isError: Boolean(meta.error) });

            // Save listings returned from RAG
            if (meta.listings && meta.listings.length) {
              const ids = new Set(meta.listings.map(l => l.id));
              setLastChatListings(allListings.filter(i => ids.has(i.id)));
            } else {
              setLastChatListings(null);
            }

          } else if (eventType === 'chunk') {
            const token = JSON.parse(dataStr);
            // Append token to the streaming assistant message
            setChatMessages(prev => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
                updated[lastIdx] = {
                  ...updated[lastIdx],
                  text: updated[lastIdx].text + token,
                };
              }
              return updated;
            });

          } else if (eventType === 'done') {
            // Mark streaming complete
            setChatMessages(prev => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
                updated[lastIdx] = { ...updated[lastIdx], streaming: false };
              }
              return updated;
            });
            break;
          }
        }
      }

    } catch (err) {
      console.error('Chat submit failed:', err);
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        text: `Lỗi kết nối RAG server: ${err.message}`,
        isError: true
      }]);
      setRagStatus({ label: 'Offline', isError: true });
    } finally {
      setIsChatLoading(false);
    }
  };

  // Focus matching listings returned from RAG chat
  const handleFocusRAGListings = () => {
    setActiveTab('search');
    setRagFilterOnly(true);
    handleFitMap();
  };

  // --- Financial / Mortgage calculations ---
  const activeListing = allListings.find(l => l.id === activeId);
  const listingPrice = activeListing?.price_vnd || 0;
  const loanPrincipal = listingPrice * (1 - downPaymentPct / 100);
  const monthlyInterestRate = (interestRate / 100) / 12;
  const loanTermMonths = loanTermYears * 12;

  let monthlyMortgagePayment = 0;
  if (monthlyInterestRate > 0 && loanPrincipal > 0) {
    monthlyMortgagePayment = loanPrincipal * (monthlyInterestRate * Math.pow(1 + monthlyInterestRate, loanTermMonths)) / (Math.pow(1 + monthlyInterestRate, loanTermMonths) - 1);
  } else if (loanPrincipal > 0) {
    monthlyMortgagePayment = loanPrincipal / loanTermMonths;
  }
  const totalInterestPayable = (monthlyMortgagePayment * loanTermMonths) - loanPrincipal;

  // Affordability Check
  let affordabilityAlert = null;
  if (monthlyIncome > 0 && monthlyMortgagePayment > 0) {
    const pct = (monthlyMortgagePayment / (monthlyIncome * 1_000_000)) * 100;
    if (pct <= 40) {
      affordabilityAlert = {
        type: 'success',
        html: `<i class="fa-solid fa-circle-check"></i> Khoản vay an toàn. Chi phí thanh toán chiếm <strong>${pct.toFixed(0)}%</strong> thu nhập hàng tháng.`
      };
    } else if (pct <= 60) {
      affordabilityAlert = {
        type: 'warning',
        html: `<i class="fa-solid fa-triangle-exclamation"></i> Cảnh báo: Chi trả chiếm <strong>${pct.toFixed(0)}%</strong> thu nhập. Khá rủi ro tài chính.`
      };
    } else {
      affordabilityAlert = {
        type: 'danger',
        html: `<i class="fa-solid fa-circle-xmark"></i> Vượt khả năng chi trả: Gốc + lãi chiếm <strong>${pct.toFixed(0)}%</strong> thu nhập.`
      };
    }
  }

  return (
    <div class="app-shell">
      {/* 1. Far-left Navigation Bar */}
      <nav class="sidebar-nav" aria-label="Menu chức năng">
        <div class="nav-top">
          <button 
            className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
            title="Trợ lý AI (Chat)"
          >
            <i class="fa-solid fa-robot"></i>
            <span>Chat</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
            title="Tìm kiếm & Bộ lọc"
          >
            <i class="fa-solid fa-magnifying-glass"></i>
            <span>Tìm kiếm</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'details' ? 'active' : ''}`}
            onClick={() => setActiveTab('details')}
            title="Chi tiết BĐS"
          >
            <i class="fa-solid fa-circle-info"></i>
            <span>Chi tiết</span>
          </button>
        </div>
        <div class="nav-bottom">
          <button 
            class="nav-item" 
            onClick={() => setTheme(prev => prev === 'dark' ? 'light' : prev === 'light' ? 'auto' : 'dark')}
            title={`Giao diện: ${theme}`}
          >
            <i class="fa-solid fa-circle-half-stroke"></i>
            <span>Giao diện</span>
          </button>
        </div>
      </nav>

      {/* 2. Sidebar Panel Content */}
      <aside class="sidebar-panel">
        
        {/* TAB 1: CHAT PANEL */}
        {activeTab === 'chat' && (
          <section id="chatTab" class="tab-content active" aria-label="Trợ lý chat">
            <header class="panel-header">
              <div>
                <h1>Trợ lý BĐS AI</h1>
                <p>RAG Agent hỗ trợ phân tích pháp lý, vị trí, tài chính & tiện ích.</p>
              </div>
              <span className={`status-pill ${ragStatus.isError ? 'error' : ''}`}>{ragStatus.label}</span>
            </header>

            <div class="suggested-prompts-container" aria-label="Gợi ý câu hỏi">
              <button type="button" class="btn-prompt" onClick={() => { setChatInput('Căn hộ 2PN dưới 3 tỷ ở TP.HCM, gần metro'); }}>Căn hộ 2PN dưới 3 tỷ ở TP.HCM, gần metro</button>
              <button type="button" class="btn-prompt" onClick={() => { setChatInput('Khu vực nào ít ngập nước, có trường học tốt?'); }}>Khu vực nào ít ngập nước, có trường học tốt?</button>
              <button type="button" class="btn-prompt" onClick={() => { setChatInput('Tôi có 3 tỷ, trả trước 30%, tính gói vay mua nhà 15 năm'); }}>Tôi có 3 tỷ, trả trước 30%, tính gói vay mua nhà 15 năm</button>
              <button type="button" class="btn-prompt" onClick={() => { setChatInput('Dự án nào có tiềm năng tăng giá quanh Thủ Đức?'); }}>Dự án nào có tiềm năng tăng giá quanh Thủ Đức?</button>
            </div>

            <div class="chat-messages-container" aria-live="polite">
              {chatMessages.map((msg, index) => (
                <article key={index} className={`message ${msg.role} ${msg.isError ? 'error' : ''} animate-fade-in`}>
                  <div class="message-header">
                    <div class="avatar">
                      {msg.role === 'user' ? <i class="fa-solid fa-user"></i> : <i class="fa-solid fa-robot"></i>}
                    </div>
                    <span class="sender-label">{msg.role === 'user' ? 'Khách hàng' : 'Trợ lý AI'}</span>
                    {msg.streaming && (
                      <span class="streaming-badge"><i class="fa-solid fa-circle-notch fa-spin"></i> Đang tạo...</span>
                    )}
                  </div>
                  <div class="message-body">
                    {msg.role === 'user' ? (
                      <p>{msg.text}</p>
                    ) : msg.text ? (
                      <div>
                        <div dangerouslySetInnerHTML={{ __html: marked.parse(msg.text) }} />
                        {msg.streaming && <span class="stream-cursor">▌</span>}
                      </div>
                    ) : msg.streaming ? (
                      <div class="typing-dots">
                        <span></span><span></span><span></span>
                      </div>
                    ) : (
                      <div dangerouslySetInnerHTML={{ __html: marked.parse(msg.text || '') }} />
                    )}

                    {msg.payload && (
                      <>
                        <div class="chat-meta-tag" title={`Intent: ${msg.payload.intent}`}>
                          {msg.payload.llm_used ? <i class="fa-solid fa-microchip"></i> : <i class="fa-solid fa-network-wired"></i>} &bull; {msg.payload.intent} &bull; {Object.keys(msg.payload.filters_applied || {}).length} filters &bull; {(msg.payload.sources || []).length} sources
                        </div>
                        
                        {msg.payload.listings && msg.payload.listings.length > 0 && (
                          <button class="btn-api" onClick={handleFocusRAGListings} style={{ marginTop: '8px', width: '100%', justifyContent: 'center' }}>
                            <i class="fa-solid fa-map-location-dot"></i> Ghim {msg.payload.listings.length} BĐS liên quan lên bản đồ
                          </button>
                        )}

                        {msg.payload.sources && msg.payload.sources.length > 0 && (
                          <>
                            <div class="sources-carousel-title" style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted-raw)', marginTop: '10px', textTransform: 'uppercase' }}>
                              <i class="fa-solid fa-book-open"></i> Tài liệu tham khảo RAG
                            </div>
                            <div class="sources-carousel">
                              {msg.payload.sources.map((source, sIdx) => {
                                let badgeLabel = "Bài viết";
                                if (source.collection === 'social_neighborhood') badgeLabel = "Ý kiến MXH";
                                else if (source.collection === 'projects') badgeLabel = "Dự án";
                                else if (source.collection === 'articles') badgeLabel = "Tin tức";

                                return (
                                  <div key={sIdx} class="source-card animate-fade-in" onClick={() => source.url && source.url !== 'None' && window.open(source.url, '_blank')} title={`Độ khớp: ${(source.score * 100).toFixed(1)}%`}>
                                    <span class="source-type-badge">{badgeLabel}</span>
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

            <form class="chat-input-form" onSubmit={handleChatSubmit}>
              <div class="input-wrapper">
                <textarea
                  id="chatInput"
                  rows={1}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Nhập câu hỏi tại đây... (Shift+Enter để xuống dòng)"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleChatSubmit();
                    }
                  }}
                />
                <button type="submit" disabled={isChatLoading} aria-label="Gửi câu hỏi">
                  {isChatLoading ? <i class="fa-solid fa-circle-notch fa-spin"></i> : <i class="fa-solid fa-paper-plane"></i>}
                </button>
              </div>
            </form>
          </section>
        )}

        {/* TAB 2: ADVANCED SEARCH & FILTERS */}
        {activeTab === 'search' && (
          <section id="searchTab" class="tab-content active" aria-label="Tìm kiếm nâng cao">
            <header class="panel-header">
              <div>
                <h2>Tìm kiếm & Bộ lọc</h2>
                <p id="evidenceSummary">Hiển thị {listings.length} bất động sản khớp</p>
              </div>
              <button type="button" class="btn-icon" onClick={handleFitMap} title="Căn chỉnh bản đồ">
                <i class="fa-solid fa-expand"></i>
              </button>
            </header>

            <div class="filter-controls-container">
              <div class="filter-group">
                <label for="searchInput">Từ khóa</label>
                <div class="input-icon-wrapper">
                  <i class="fa-solid fa-magnifying-glass"></i>
                  <input 
                    id="searchInput" 
                    type="search" 
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Tên dự án, đường, quận..." 
                  />
                </div>
              </div>

              <div class="filter-row-grid">
                <div class="filter-group">
                  <label for="listingType">Giao dịch</label>
                  <select id="listingType" value={listingType} onChange={(e) => setListingType(e.target.value)}>
                    <option value="">Tất cả</option>
                    <option value="ban">Mua bán</option>
                    <option value="cho-thue">Cho thuê</option>
                  </select>
                </div>
                <div class="filter-group">
                  <label for="provinceSelect">Tỉnh / Thành</label>
                  <select id="provinceSelect" value={provinceSelect} onChange={(e) => setProvinceSelect(e.target.value)}>
                    <option value="">Tất cả</option>
                    {provinces.map(p => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
              </div>

              <div class="filter-group">
                <div class="filter-label-row">
                  <span>Khoảng giá</span>
                  <span class="range-value-display">
                    {minPrice || maxPrice ? `${minPrice || 0} - ${maxPrice || '∞'} Tỷ` : 'Tất cả'}
                  </span>
                </div>
                <div class="range-inputs">
                  <input type="number" placeholder="Min (Tỷ)" min="0" step="0.1" value={minPrice} onChange={(e) => setMinPrice(e.target.value)} />
                  <input type="number" placeholder="Max (Tỷ)" min="0" step="0.1" value={maxPrice} onChange={(e) => setMaxPrice(e.target.value)} />
                </div>
              </div>

              <div class="filter-group">
                <div class="filter-label-row">
                  <span>Diện tích (m²)</span>
                  <span class="range-value-display">
                    {minArea || maxArea ? `${minArea || 0} - ${maxArea || '∞'} m²` : 'Tất cả'}
                  </span>
                </div>
                <div class="range-inputs">
                  <input type="number" placeholder="Min" min="0" value={minArea} onChange={(e) => setMinArea(e.target.value)} />
                  <input type="number" placeholder="Max" min="0" value={maxArea} onChange={(e) => setMaxArea(e.target.value)} />
                </div>
              </div>

              <div class="filter-row-grid">
                <div class="filter-group">
                  <label for="bedsSelect">Phòng ngủ</label>
                  <select id="bedsSelect" value={bedsSelect} onChange={(e) => setBedsSelect(e.target.value)}>
                    <option value="">Tất cả</option>
                    <option value="1">1 PN</option>
                    <option value="2">2 PN</option>
                    <option value="3">3 PN</option>
                    <option value="4">4+ PN</option>
                  </select>
                </div>
                <div class="filter-group">
                  <label for="sortBySelect">Sắp xếp</label>
                  <select id="sortBySelect" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                    <option value="">Mặc định</option>
                    <option value="price_asc">Giá: Thấp đến Cao</option>
                    <option value="price_desc">Giá: Cao đến Thấp</option>
                    <option value="area_desc">Diện tích: Lớn đến Nhỏ</option>
                  </select>
                </div>
              </div>

              {lastChatListings && (
                <div class="rag-toggle-container">
                  <label class="switch-label">
                    <input type="checkbox" checked={ragFilterOnly} onChange={(e) => setRagFilterOnly(e.target.checked)} />
                    <span class="slider-switch"></span>
                    <span>Chỉ hiện tin từ kết quả RAG gần nhất</span>
                  </label>
                </div>
              )}
            </div>

            <div class="stats-row" aria-label="Thống kê">
              <div class="stat-card">
                <span class="stat-val">{listings.length}</span>
                <span class="stat-lbl">Tổng tin đăng</span>
              </div>
              <div class="stat-card">
                <span class="stat-val text-sale">{listings.filter(i => i.listing_type === 'ban').length}</span>
                <span class="stat-lbl">Tin bán</span>
              </div>
              <div class="stat-card">
                <span class="stat-val text-rent">{listings.filter(i => i.listing_type === 'cho-thue').length}</span>
                <span class="stat-lbl">Tin thuê</span>
              </div>
            </div>

            <div id="listingList" class="listing-list-scroll">
              {listings.length === 0 ? (
                <div class="poi-empty">
                  <i class="fa-solid fa-house-circle-xmark" style={{ fontSize: '24px', display: 'block', marginBottom: '8px' }}></i>
                  Không có tin đăng nào khớp với bộ lọc của bạn.
                </div>
              ) : (
                listings.map(item => (
                  <article 
                    key={item.id}
                    className={`listing-card ${item.id === activeId ? 'active' : ''}`}
                    onClick={() => handleSelectListing(item.id)}
                  >
                    {item.image ? (
                      <img src={item.image} alt={item.title} loading="lazy" />
                    ) : (
                      <div style={{ width: '90px', height: '80px', backgroundColor: 'var(--border-raw)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted-raw)', flexShrink: 0 }}>
                        <i class="fa-solid fa-image"></i>
                      </div>
                    )}
                    <div class="listing-info">
                      <h3>{item.title}</h3>
                      <div class="listing-meta-tags">
                        <span className={`badge-tag ${item.listing_type === 'ban' ? 'sale' : 'rent'}`}>
                          {item.listing_type === 'ban' ? 'Bán' : 'Thuê'}
                        </span>
                        <span class="badge-tag price">{formatMoney(item.price_vnd)}</span>
                        {item.area_m2 && <span class="badge-tag">{item.area_m2} m²</span>}
                      </div>
                      <p class="listing-address"><i class="fa-solid fa-location-dot"></i> {item.address}</p>
                    </div>
                  </article>
                ))
              )}
            </div>
          </section>
        )}

        {/* TAB 3: DETAILS & FINANCIAL CALCULATOR */}
        {activeTab === 'details' && (
          <section id="detailsTab" class="tab-content active" aria-label="Chi tiết bất động sản">
            {!activeId || !activeListing ? (
              <div class="details-placeholder animate-fade-in">
                <i class="fa-solid fa-house-circle-exclamation"></i>
                <h3>Chưa chọn bất động sản</h3>
                <p>Nhấp vào một tin đăng trong danh sách hoặc ghim trên bản đồ để xem chi tiết, tiện ích xung quanh và tính toán tài chính.</p>
              </div>
            ) : (
              <div class="details-content-scroll">
                <div>
                  <div class="detail-image-wrapper">
                    {activeListing.image ? (
                      <img src={activeListing.image} alt={activeListing.title} />
                    ) : (
                      <div style={{ height: '180px', backgroundColor: 'var(--border-raw)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted-raw)' }}>
                        <i class="fa-solid fa-image" style={{ fontSize: '32px' }}></i>
                      </div>
                    )}
                  </div>
                  
                  <div style={{ marginTop: '14px' }}>
                    <h3 class="detail-title">{activeListing.title}</h3>
                    <p class="detail-location-text"><i class="fa-solid fa-location-dot"></i> {activeListing.address}</p>
                    
                    <div class="detail-params-row">
                      <div class="param-badge">
                        <span class="param-val">{formatMoney(activeListing.price_vnd)}</span>
                        <span class="param-lbl">Giá cả</span>
                      </div>
                      <div class="param-badge">
                        <span class="param-val">{activeListing.area_m2 ? `${activeListing.area_m2} m²` : 'N/A'}</span>
                        <span class="param-lbl">Diện tích</span>
                      </div>
                      <div class="param-badge">
                        <span class="param-val">{activeListing.bedrooms ? `${activeListing.bedrooms} PN` : 'N/A'}</span>
                        <span class="param-lbl">Phòng ngủ</span>
                      </div>
                    </div>

                    <div class="detail-description-section" style={{ marginTop: '16px' }}>
                      <h4>Mô tả tin đăng</h4>
                      <p class="detail-description-text">{activeListing.title} - Toạ lạc tại khu vực {activeListing.district}, {activeListing.province}. Bất động sản này sở hữu đầy đủ tiềm năng đầu tư, môi trường sống xanh và kết nối hạ tầng giao thông lý tưởng.</p>
                    </div>

                    <div class="detail-meta-table">
                      <div class="meta-cell"><span class="meta-lbl">Pháp lý:</span> <span class="meta-val">{activeListing.legal || 'Đang cập nhật'}</span></div>
                      <div class="meta-cell"><span class="meta-lbl">Nội thất:</span> <span class="meta-val">{activeListing.furniture || 'Đang cập nhật'}</span></div>
                      <div class="meta-cell"><span class="meta-lbl">Giao dịch:</span> <span class="meta-val">{activeListing.listing_type === 'ban' ? 'Mua bán' : 'Cho thuê'}</span></div>
                      <div class="meta-cell"><span class="meta-lbl">Phòng tắm:</span> <span class="meta-val">{activeListing.bathrooms || 'N/A'} WC</span></div>
                    </div>
                    
                    {activeListing.url && (
                      <a href={activeListing.url} target="_blank" rel="noreferrer" class="btn-api" style={{ marginTop: '14px', width: '100%', justifyContent: 'center' }}>
                        <i class="fa-solid fa-arrow-up-right-from-square"></i> Xem liên kết tin gốc
                      </a>
                    )}
                  </div>
                </div>

                {/* Nearby POIs */}
                <section class="amenities-section card-glass">
                  <h3><i class="fa-solid fa-map-location-dot"></i> Tiện ích lân cận (Bán kính 2km)</h3>
                  <div class="pois-grid">
                    {isPoisLoading ? (
                      <div class="poi-loading"><i class="fa-solid fa-circle-notch fa-spin"></i> Đang tải tiện ích...</div>
                    ) : pois.length === 0 ? (
                      <div class="poi-empty"><i class="fa-solid fa-house-circle-exclamation"></i> Không có tiện ích lân cận nào được lưu.</div>
                    ) : (
                      pois.map((poi, pIdx) => {
                        let icon = '<i class="fa-solid fa-location-dot"></i>';
                        if (poi.category === 'transit_station') icon = '<i class="fa-solid fa-train-subway"></i>';
                        else if (poi.category === 'school') icon = '<i class="fa-solid fa-graduation-cap"></i>';
                        else if (poi.category === 'hospital') icon = '<i class="fa-solid fa-house-medical"></i>';
                        else if (poi.category === 'park') icon = '<i class="fa-solid fa-tree"></i>';

                        return (
                          <div key={pIdx} className={`poi-item ${poi.category}`} title={poi.address}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                              <span style={{ color: 'var(--text-muted-raw)', fontSize: '13px' }} dangerouslySetInnerHTML={{ __html: icon }} />
                              <span class="poi-name">{poi.name}</span>
                            </div>
                            <span class="poi-dist">{poi.distance_m}m</span>
                          </div>
                        );
                      })
                    )}
                  </div>
                </section>

                {/* Mortgage loan calculator */}
                <section class="mortgage-section card-glass">
                  <h3><i class="fa-solid fa-calculator"></i> Tính toán tài chính vay mua nhà</h3>
                  <div class="calculator-form">
                    <div class="calc-group">
                      <div class="calc-label-row">
                        <label>Tỷ lệ trả trước</label>
                        <span>{downPaymentPct}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="10" 
                        max="90" 
                        step="5" 
                        value={downPaymentPct} 
                        onChange={(e) => setDownPaymentPct(parseInt(e.target.value))} 
                      />
                    </div>

                    <div class="calc-row">
                      <div class="calc-group">
                        <label>Lãi suất (% / năm)</label>
                        <input 
                          type="number" 
                          min="1" 
                          max="25" 
                          step="0.1" 
                          value={interestRate} 
                          onChange={(e) => setInterestRate(parseFloat(e.target.value) || 0)} 
                        />
                      </div>
                      <div class="calc-group">
                        <label>Kỳ hạn vay</label>
                        <select value={loanTermYears} onChange={(e) => setLoanTermYears(parseInt(e.target.value))}>
                          <option value="5">5 năm</option>
                          <option value="10">10 năm</option>
                          <option value="15">15 năm</option>
                          <option value="20">20 năm</option>
                          <option value="25">25 năm</option>
                          <option value="30">30 năm</option>
                        </select>
                      </div>
                    </div>

                    <div class="calc-group">
                      <label>Thu nhập hàng tháng (Triệu VNĐ)</label>
                      <input 
                        type="number" 
                        min="0" 
                        value={monthlyIncome} 
                        onChange={(e) => setMonthlyIncome(e.target.value)} 
                        placeholder="Nhập thu nhập để đánh giá..." 
                      />
                    </div>

                    <div class="calc-results-card">
                      <div class="result-row">
                        <span>Giá trị BĐS:</span>
                        <strong>{formatMoney(listingPrice)}</strong>
                      </div>
                      <div class="result-row">
                        <span>Số tiền trả trước:</span>
                        <span>{formatMoney(listingPrice * (downPaymentPct / 100))}</span>
                      </div>
                      <div class="result-row">
                        <span>Số tiền cần vay:</span>
                        <strong class="text-accent">{formatMoney(loanPrincipal)}</strong>
                      </div>
                      <hr class="calc-divider" />
                      <div class="result-row highlight">
                        <span>Gốc + Lãi tháng đầu:</span>
                        <strong>{listingPrice > 0 ? `${formatMoney(monthlyMortgagePayment)} / tháng` : '--'}</strong>
                      </div>
                      <div class="result-row">
                        <span>Tổng tiền lãi phải trả:</span>
                        <span>{listingPrice > 0 ? formatMoney(totalInterestPayable) : '--'}</span>
                      </div>
                      
                      {affordabilityAlert && (
                        <div 
                          className={`affordability-alert ${affordabilityAlert.type} animate-fade-in`}
                          dangerouslySetInnerHTML={{ __html: affordabilityAlert.html }}
                        />
                      )}
                    </div>
                  </div>
                </section>
              </div>
            )}
          </section>
        )}
      </aside>

      {/* 3. Main Map Area */}
      <main class="map-area">
        <div class="map-overlay-toolbar card-glass animate-slide-down">
          <div class="toolbar-left">
            <span class="app-logo"><i class="fa-solid fa-compass-drafting"></i> Maps Portal</span>
            <span id="mapTitle">Bản đồ phân phối ({listings.length} ghim)</span>
          </div>
          <div class="toolbar-right">
            <span id="geoNote" class="geo-badge"><i class="fa-solid fa-location-crosshairs"></i> Vị trí ước lượng</span>
            <a class="btn-api" href="/docs" target="_blank" rel="noreferrer" title="Swagger API Documentation"><i class="fa-solid fa-code"></i> API Docs</a>
          </div>
        </div>
        <div id="map"></div>
      </main>
    </div>
  );
}
