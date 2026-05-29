# Real Estate UI Website

Static RAG chatbot + real estate map UI. The page is intentionally focused: chat on the left, map with property markers on the right.

## Files

- `index.html` - two-column chatbot/map layout
- `styles.css` - responsive app styling
- `script.js` - RAG chat logic, local retrieval, map markers, property focus
- `rag/context.sample.js` - sample RAG context loaded by the chatbot
- `rag/rag-api-template.py` - optional FastAPI template for real RAG later

## Run

Open `index.html` in your browser, or serve the folder with any static server.

## RAG Notes

The chatbot currently works locally with sample context from `rag/context.sample.js`.
To add your own context, edit `window.RAG_CONTEXT_DOCS` in that file. Property documents in `script.js` can include `propertyId`, so a chat answer can zoom the map to a marker.

To connect a real backend later, run your RAG API and set `RAG_API_URL` in
`script.js`, for example:

```js
const RAG_API_URL = "http://localhost:8000/chat";
```
