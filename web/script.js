const properties = [
  {
    id: 1,
    title: "Thu Thiem Skyline Apartment",
    type: "Apartment",
    city: "Ho Chi Minh City",
    address: "Thu Thiem, Thu Duc City, Ho Chi Minh City",
    price: 820000,
    beds: 2,
    baths: 2,
    area: 1240,
    lat: 10.7769,
    lng: 106.7286,
    description: "Bright apartment near Thu Thiem with skyline views, concierge service, coworking lounge, and quick access to District 1."
  },
  {
    id: 2,
    title: "Thao Dien Garden Villa",
    type: "Villa",
    city: "Ho Chi Minh City",
    address: "Thao Dien, Thu Duc City, Ho Chi Minh City",
    price: 1250000,
    beds: 5,
    baths: 4,
    area: 3420,
    lat: 10.8022,
    lng: 106.7338,
    description: "Private villa in Thao Dien with garden, pool, guest suite, outdoor kitchen, and quiet residential surroundings."
  },
  {
    id: 3,
    title: "Phu My Hung Family House",
    type: "House",
    city: "Ho Chi Minh City",
    address: "Phu My Hung, District 7, Ho Chi Minh City",
    price: 745000,
    beds: 4,
    baths: 3,
    area: 2680,
    lat: 10.7294,
    lng: 106.7217,
    description: "Modern family house near international schools, parks, shopping, with flexible office space and a calm neighborhood."
  },
  {
    id: 4,
    title: "Ben Thanh Boutique Condo",
    type: "Condo",
    city: "Ho Chi Minh City",
    address: "Ben Thanh, District 1, Ho Chi Minh City",
    price: 1180000,
    beds: 2,
    baths: 2,
    area: 1385,
    lat: 10.7721,
    lng: 106.6983,
    description: "Polished condo in District 1 with designer kitchen, city access, storage, security, and restaurants nearby."
  },
  {
    id: 5,
    title: "Binh Thanh Starter Home",
    type: "House",
    city: "Ho Chi Minh City",
    address: "Binh Thanh District, Ho Chi Minh City",
    price: 515000,
    beds: 2,
    baths: 1,
    area: 980,
    lat: 10.8033,
    lng: 106.6967,
    description: "Renovated compact home in Binh Thanh with practical layout, quick access to District 1, and low maintenance cost."
  },
  {
    id: 6,
    title: "Sala Premium Apartment",
    type: "Apartment",
    city: "Ho Chi Minh City",
    address: "Sala, Thu Duc City, Ho Chi Minh City",
    price: 895000,
    beds: 3,
    baths: 2,
    area: 1840,
    lat: 10.7705,
    lng: 106.7381,
    description: "Premium apartment in Sala with smart-home automation, secure parking, green streets, and river access."
  }
];

const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const chatMessages = document.querySelector("#chatMessages");
const activePropertyTitle = document.querySelector("#activePropertyTitle");
const activePropertyMeta = document.querySelector("#activePropertyMeta");

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0
});

const defaultRagDocuments = properties.map((property) => ({
  id: `property-${property.id}`,
  title: property.title,
  type: "Listing",
  propertyId: property.id,
  content: `${property.title} is a ${property.type} in ${property.city}. Price: ${money.format(property.price)}. Bedrooms: ${property.beds}. Bathrooms: ${property.baths}. Area: ${property.area} square feet. Address: ${property.address}. ${property.description}`
}));

const guideDocuments = [
  {
    id: "family-guide",
    title: "Family home guide",
    type: "Guide",
    content: "For families, prioritize three or more bedrooms, schools, parks, safe neighborhoods, office space, and outdoor areas."
  },
  {
    id: "investment-guide",
    title: "Investment guide",
    type: "Guide",
    content: "For investment buyers, compare purchase price, rental appeal, transit access, maintenance level, city demand, and long-term growth."
  },
  {
    id: "tour-policy",
    title: "Tour booking policy",
    type: "Policy",
    content: "Users can request a property tour by providing name, email, preferred property, and visit time. Agents confirm within 24 hours."
  }
];

const externalRagDocuments = Array.isArray(window.RAG_CONTEXT_DOCS) ? window.RAG_CONTEXT_DOCS : [];
const ragDocuments = [...externalRagDocuments, ...defaultRagDocuments, ...guideDocuments];
const RAG_API_URL = "";

let map;
const markers = new Map();

function tokenize(text) {
  const stopWords = new Set(["the", "and", "for", "with", "that", "this", "from", "have", "has", "can", "need", "want", "toi", "can", "nha", "cho"]);
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2 && !stopWords.has(word));
}

function retrieveContext(query, limit = 5) {
  const queryTerms = tokenize(query);
  return ragDocuments
    .map((document) => {
      const text = `${document.title} ${document.type} ${document.content}`;
      const documentTerms = tokenize(text);
      const score = queryTerms.reduce((total, term) => {
        const exactMatches = documentTerms.filter((word) => word === term).length;
        const partialMatch = text.toLowerCase().includes(term) ? 1 : 0;
        return total + exactMatches + partialMatch;
      }, 0);
      return { ...document, score };
    })
    .filter((document) => document.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function getPropertyFromContext(context) {
  const directListing = context.find((document) => document.propertyId);
  if (directListing) {
    return properties.find((property) => property.id === directListing.propertyId);
  }

  const listingByTitle = context.find((document) =>
    properties.some((property) => document.title.toLowerCase().includes(property.title.toLowerCase()))
  );
  if (!listingByTitle) return null;

  return properties.find((property) => listingByTitle.title.toLowerCase().includes(property.title.toLowerCase()));
}

function generateRagAnswer(question, context) {
  if (!context.length) {
    return {
      answer:
        "I do not have enough matching RAG context for that yet. Add more documents in rag/context.sample.js, then ask again.",
      sources: [],
      property: null
    };
  }

  const property = getPropertyFromContext(context);
  const sources = context.map((document, index) => `[${index + 1}] ${document.title}`);

  let answer = "Based on your RAG context, ";
  if (property) {
    answer += `${property.title} looks relevant. It is a ${property.type} in ${property.city}, priced at ${money.format(property.price)}, with ${property.beds} bedrooms, ${property.baths} bathrooms, and ${property.area.toLocaleString()} sqft. ${property.description}`;
  } else {
    answer += `${context[0].title} is the most relevant source. ${context[0].content}`;
  }

  answer += `\n\nSources: ${sources.join("; ")}.`;
  return { answer, sources, property };
}

async function askRealRagApi(question) {
  if (!RAG_API_URL) return null;

  const response = await fetch(RAG_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      conversation_id: localStorage.getItem("rag_conversation_id")
    })
  });

  if (!response.ok) throw new Error("RAG API request failed");

  const data = await response.json();
  if (data.conversation_id) {
    localStorage.setItem("rag_conversation_id", data.conversation_id);
  }

  return {
    answer: data.answer,
    sources: (data.sources || []).map((source, index) => `[${index + 1}] ${source.title}`),
    property: properties.find((property) => property.id === data.property_id) || null
  };
}

async function answerQuestion(question) {
  const apiAnswer = await askRealRagApi(question);
  if (apiAnswer) return apiAnswer;

  const context = retrieveContext(question);
  return generateRagAnswer(question, context);
}

function addMessage(role, text, sources = []) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  if (sources.length) {
    const small = document.createElement("small");
    small.textContent = sources.join(" | ");
    message.appendChild(small);
  }
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return message;
}

async function streamMessage(element, text, sources = []) {
  element.textContent = "";
  const words = text.split(" ");
  for (const word of words) {
    element.textContent += `${word} `;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    await new Promise((resolve) => setTimeout(resolve, 22));
  }
  if (sources.length) {
    const small = document.createElement("small");
    small.textContent = sources.join(" | ");
    element.appendChild(small);
  }
}

function setActiveProperty(property) {
  activePropertyTitle.textContent = property.title;
  activePropertyMeta.textContent = `${property.city} | ${money.format(property.price)} | ${property.beds} beds`;
}

function focusProperty(property) {
  if (!property || !map) return;
  const marker = markers.get(property.id);
  map.setView([property.lat, property.lng], 12, { animate: true });
  setActiveProperty(property);
  if (marker) marker.openPopup();
}

function initMap() {
  if (!window.L) {
    document.querySelector("#map").innerHTML = "<p class='map-fallback'>Map library could not load.</p>";
    return;
  }

  map = L.map("map", {
    zoomControl: true,
    scrollWheelZoom: true
  }).setView([10.7769, 106.7009], 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap"
  }).addTo(map);

  properties.forEach((property) => {
    const marker = L.marker([property.lat, property.lng]).addTo(map);
    marker.bindPopup(`
      <div class="property-popup">
        <strong>${property.title}</strong>
        <span>${property.city} | ${money.format(property.price)}</span>
        <span>${property.beds} beds | ${property.baths} baths | ${property.area.toLocaleString()} sqft</span>
      </div>
    `);
    marker.on("click", () => setActiveProperty(property));
    markers.set(property.id, marker);
  });
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  chatInput.value = "";
  addMessage("user", question);
  const assistantMessage = addMessage("assistant", "Searching your RAG context...");

  try {
    const { answer, sources, property } = await answerQuestion(question);
    await streamMessage(assistantMessage, answer, sources);
    if (property) focusProperty(property);
  } catch (error) {
    await streamMessage(
      assistantMessage,
      "The RAG service is not available. Check RAG_API_URL, or continue with local context from rag/context.sample.js."
    );
  }
});

initMap();
addMessage(
  "assistant",
  "Ask me about properties, budget, family fit, investment potential, or your own RAG context. Matching properties will be highlighted on the map."
);
