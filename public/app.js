/* fab-search — Frontend application logic */

(function () {
    "use strict";

    // ── DOM refs ──────────────────────────────────────────────
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const cameraInput = document.getElementById("camera-input");
    const cameraBtn = document.getElementById("camera-btn");
    const galleryBtn = document.getElementById("gallery-btn");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const clearBtn = document.getElementById("clear-btn");
    const topKSlider = document.getElementById("top-k");
    const topKValue = document.getElementById("top-k-value");
    const searchBtn = document.getElementById("search-btn");
    const loading = document.getElementById("loading");
    const resultsContainer = document.getElementById("results");

    const cropModal = document.getElementById("crop-modal");
    const cropImage = document.getElementById("crop-image");
    const cropConfirmBtn = document.getElementById("crop-confirm-btn");
    const cropCancelBtn = document.getElementById("crop-cancel-btn");
    const recropBtn = document.getElementById("recrop-btn");

    const catalogsLoading = document.getElementById("catalogs-loading");
    const catalogsList = document.getElementById("catalogs-list");
    const catalogsEmpty = document.getElementById("catalogs-empty");

    // Catalog filter DOM refs
    const catalogFilterSummary = document.getElementById("catalog-filter-summary");
    const catalogFilterToggle = document.getElementById("catalog-filter-toggle");
    const catalogFilterBody = document.getElementById("catalog-filter-body");
    const catalogPillsContainer = document.getElementById("catalog-pills");
    const catalogSelectAll = document.getElementById("catalog-select-all");
    const catalogSelectNone = document.getElementById("catalog-select-none");
    const templateSelect = document.getElementById("template-select");
    const templateSaveBtn = document.getElementById("template-save-btn");
    const templateDeleteBtn = document.getElementById("template-delete-btn");

    let selectedFile = null;
    let originalDataUrl = null;
    let cropper = null;

    // Catalog filter state
    let allCatalogNames = [];
    let selectedCatalogs = new Set();
    const TEMPLATE_KEY = "fab-search-collections";

    // ── Navigation ────────────────────────────────────────────
    document.querySelectorAll(".nav-links a").forEach((link) => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
            document.querySelectorAll(".nav-links a").forEach((a) => a.classList.remove("active"));
            document.getElementById("page-" + page).classList.add("active");
            link.classList.add("active");

            if (page === "catalogs") loadCatalogs();
            if (page === "pages") loadPagesCatalogList();
        });
    });

    // ── Catalog filter ────────────────────────────────────────
    catalogFilterToggle.addEventListener("click", () => {
        catalogFilterBody.classList.toggle("hidden");
        catalogFilterToggle.textContent = catalogFilterBody.classList.contains("hidden") ? "Change" : "Hide";
    });

    catalogSelectAll.addEventListener("click", () => {
        selectedCatalogs = new Set(allCatalogNames);
        renderCatalogPills();
    });

    catalogSelectNone.addEventListener("click", () => {
        selectedCatalogs.clear();
        renderCatalogPills();
    });

    function loadCatalogFilter() {
        fetch("/api/catalogs")
            .then((r) => r.json())
            .then((data) => {
                allCatalogNames = data.catalogs.map((c) => c.pdf_name);
                selectedCatalogs = new Set(allCatalogNames);
                renderCatalogPills();
                renderTemplateDropdown();
            })
            .catch(() => {});
    }

    function renderCatalogPills() {
        catalogPillsContainer.innerHTML = allCatalogNames
            .map((name) => {
                var cls = selectedCatalogs.has(name) ? "catalog-pill selected" : "catalog-pill";
                return '<span class="' + cls + '" data-catalog="' + escapeHtml(name) + '">' +
                    escapeHtml(name.replace(/\.pdf$/i, "")) + "</span>";
            })
            .join("");

        catalogPillsContainer.querySelectorAll(".catalog-pill").forEach((pill) => {
            pill.addEventListener("click", () => {
                var name = pill.dataset.catalog;
                if (selectedCatalogs.has(name)) {
                    selectedCatalogs.delete(name);
                } else {
                    selectedCatalogs.add(name);
                }
                pill.classList.toggle("selected");
                updateFilterSummary();
            });
        });

        updateFilterSummary();
    }

    function updateFilterSummary() {
        if (selectedCatalogs.size === 0) {
            catalogFilterSummary.textContent = "None selected";
        } else if (selectedCatalogs.size === allCatalogNames.length) {
            catalogFilterSummary.textContent = "All catalogs";
        } else {
            catalogFilterSummary.textContent = selectedCatalogs.size + " of " + allCatalogNames.length + " catalogs";
        }
    }

    // ── Collections ───────────────────────────────────────────
    function getTemplates() {
        try {
            return JSON.parse(localStorage.getItem(TEMPLATE_KEY)) || [];
        } catch (_) {
            return [];
        }
    }

    function saveTemplates(templates) {
        localStorage.setItem(TEMPLATE_KEY, JSON.stringify(templates));
    }

    function renderTemplateDropdown() {
        var templates = getTemplates();
        templateSelect.innerHTML = '<option value="">Load collection...</option>';
        templates.forEach(function (t, i) {
            var opt = document.createElement("option");
            opt.value = i;
            opt.textContent = t.name;
            templateSelect.appendChild(opt);
        });
    }

    templateSelect.addEventListener("change", () => {
        var idx = templateSelect.value;
        if (idx === "") return;
        var templates = getTemplates();
        var t = templates[+idx];
        if (!t) return;
        selectedCatalogs = new Set(t.catalogs.filter((c) => allCatalogNames.includes(c)));
        renderCatalogPills();
    });

    templateSaveBtn.addEventListener("click", () => {
        if (selectedCatalogs.size === 0) return;
        var name = prompt("Collection name:");
        if (!name || !name.trim()) return;
        var templates = getTemplates();
        templates.push({ name: name.trim(), catalogs: Array.from(selectedCatalogs) });
        saveTemplates(templates);
        renderTemplateDropdown();
    });

    templateDeleteBtn.addEventListener("click", () => {
        var idx = templateSelect.value;
        if (idx === "") return;
        var templates = getTemplates();
        var t = templates[+idx];
        if (!t) return;
        if (!confirm('Delete collection "' + t.name + '"?')) return;
        templates.splice(+idx, 1);
        saveTemplates(templates);
        renderTemplateDropdown();
    });

    // Load catalog filter on page init
    loadCatalogFilter();

    // ── File upload ───────────────────────────────────────────
    cameraBtn.addEventListener("click", (e) => { e.stopPropagation(); cameraInput.click(); });
    galleryBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    cameraInput.addEventListener("change", () => {
        if (cameraInput.files.length) handleFile(cameraInput.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            originalDataUrl = e.target.result;
            openCropModal(originalDataUrl);
        };
        reader.readAsDataURL(file);
    }

    function openCropModal(dataUrl) {
        cropImage.src = dataUrl;
        cropModal.classList.remove("hidden");
        uploadArea.classList.add("hidden");
        previewContainer.classList.add("hidden");

        if (cropper) cropper.destroy();
        cropper = new Cropper(cropImage, {
            viewMode: 1,
            autoCropArea: 0.8,
            movable: true,
            zoomable: true,
            rotatable: false,
            scalable: false,
            responsive: true,
            background: false,
            checkCrossOrigin: false,
        });
    }

    cropConfirmBtn.addEventListener("click", () => {
        if (!cropper) return;
        const canvas = cropper.getCroppedCanvas();
        canvas.toBlob((blob) => {
            selectedFile = new File([blob], "crop.jpg", { type: "image/jpeg" });
            previewImage.src = URL.createObjectURL(blob);
            previewContainer.classList.remove("hidden");
            cropModal.classList.add("hidden");
            cropper.destroy();
            cropper = null;
            searchBtn.disabled = false;
        }, "image/jpeg", 0.9);
    });

    cropCancelBtn.addEventListener("click", () => {
        if (cropper) { cropper.destroy(); cropper = null; }
        cropModal.classList.add("hidden");
        originalDataUrl = null;
        selectedFile = null;
        fileInput.value = "";
        cameraInput.value = "";
        uploadArea.classList.remove("hidden");
    });

    recropBtn.addEventListener("click", () => {
        if (originalDataUrl) openCropModal(originalDataUrl);
    });

    clearBtn.addEventListener("click", () => {
        if (cropper) { cropper.destroy(); cropper = null; }
        selectedFile = null;
        originalDataUrl = null;
        fileInput.value = "";
        cameraInput.value = "";
        previewContainer.classList.add("hidden");
        cropModal.classList.add("hidden");
        uploadArea.classList.remove("hidden");
        searchBtn.disabled = true;
        resultsContainer.innerHTML = "";
    });

    // ── Top-K slider ──────────────────────────────────────────
    topKSlider.addEventListener("input", () => {
        topKValue.textContent = topKSlider.value;
    });

    // ── Search ────────────────────────────────────────────────
    searchBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        searchBtn.disabled = true;
        loading.classList.remove("hidden");
        resultsContainer.innerHTML = "";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const topK = topKSlider.value;
            var url = "/api/search?top_k=" + encodeURIComponent(topK);
            if (selectedCatalogs.size > 0 && selectedCatalogs.size < allCatalogNames.length) {
                url += "&catalogs=" + encodeURIComponent(Array.from(selectedCatalogs).join(","));
            }
            const res = await fetch(url, {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json().catch(() => null);
                throw new Error(err?.detail || "Search failed (" + res.status + ")");
            }

            const data = await res.json();
            renderResults(data.results);
        } catch (err) {
            resultsContainer.innerHTML =
                '<p class="empty-state">Error: ' + escapeHtml(err.message) + "</p>";
        } finally {
            loading.classList.add("hidden");
            searchBtn.disabled = false;
        }
    });

    function renderResults(results) {
        if (!results.length) {
            resultsContainer.innerHTML = '<p class="empty-state">No matches found.</p>';
            return;
        }

        resultsContainer.innerHTML = results
            .map(
                (r) =>
                    '<div class="result-card">' +
                    '  <div class="result-image-wrap">' +
                    '    <img src="' + escapeHtml(r.page_image) + '" alt="Page ' + r.page_number + '"' +
                    '         data-tile-x="' + r.tile_x + '" data-tile-y="' + r.tile_y + '"' +
                    '         data-tile-w="' + r.tile_w + '" data-tile-h="' + r.tile_h + '"' +
                    '         onload="drawHighlight(this)">' +
                    '    <canvas></canvas>' +
                    '    <span class="result-rank">' + r.rank + "</span>" +
                    '    <span class="result-score">' + (r.score * 100).toFixed(1) + "%</span>" +
                    "  </div>" +
                    '  <div class="result-meta">' +
                    '    <div><span class="label">Catalog</span><br><span class="value">' +
                    escapeHtml(r.pdf_name) + "</span></div>" +
                    '    <div><span class="label">Page</span><br><span class="value">' +
                    r.page_number + "</span></div>" +
                    "  </div>" +
                    "</div>"
            )
            .join("");
    }

    // ── Image viewer modal ────────────────────────────────────
    const viewerModal = document.getElementById("viewer-modal");
    const viewerImage = document.getElementById("viewer-image");
    const viewerCanvas = document.getElementById("viewer-canvas");
    const viewerBody = document.getElementById("viewer-body");
    const viewerCloseBtn = document.getElementById("viewer-close-btn");
    const viewerMeta = document.getElementById("viewer-meta");

    resultsContainer.addEventListener("click", (e) => {
        const wrap = e.target.closest(".result-image-wrap");
        if (!wrap) return;
        const img = wrap.querySelector("img");
        if (!img) return;

        viewerImage.src = img.src;
        viewerImage.onload = () => {
            viewerCanvas.width = viewerImage.naturalWidth;
            viewerCanvas.height = viewerImage.naturalHeight;
            const tileX = +img.dataset.tileX;
            const tileY = +img.dataset.tileY;
            const tileW = +img.dataset.tileW;
            const tileH = +img.dataset.tileH;
            const natW = viewerImage.naturalWidth;
            const natH = viewerImage.naturalHeight;
            if (!(tileX === 0 && tileY === 0 && tileW === natW && tileH === natH)) {
                const ctx = viewerCanvas.getContext("2d");
                ctx.clearRect(0, 0, viewerCanvas.width, viewerCanvas.height);
                ctx.strokeStyle = "#ef4444";
                ctx.lineWidth = 4;
                ctx.strokeRect(tileX, tileY, tileW, tileH);
            }
        };

        const card = wrap.closest(".result-card");
        const meta = card ? card.querySelector(".result-meta") : null;
        viewerMeta.textContent = meta ? meta.textContent.replace(/\s+/g, " ").trim() : "";

        viewerState.scale = 1;
        viewerState.x = 0;
        viewerState.y = 0;
        applyViewerTransform();
        viewerModal.classList.remove("hidden");
    });

    viewerCloseBtn.addEventListener("click", () => {
        viewerModal.classList.add("hidden");
    });

    // Close on backdrop click (not on the image itself)
    viewerModal.addEventListener("click", (e) => {
        if (e.target === viewerModal || e.target.classList.contains("viewer-content")) {
            viewerModal.classList.add("hidden");
        }
    });

    // ── Viewer pan & zoom ─────────────────────────────────────
    const viewerState = { scale: 1, x: 0, y: 0 };
    const pointers = new Map();
    let lastPinchDist = 0;
    let lastPinchCenter = null;
    let lastTapTime = 0;

    function applyViewerTransform() {
        viewerBody.style.transform =
            "translate(" + viewerState.x + "px," + viewerState.y + "px) scale(" + viewerState.scale + ")";
    }

    function getDistance(p1, p2) {
        const dx = p1.clientX - p2.clientX;
        const dy = p1.clientY - p2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function getCenter(p1, p2) {
        return { x: (p1.clientX + p2.clientX) / 2, y: (p1.clientY + p2.clientY) / 2 };
    }

    const viewerContent = document.querySelector(".viewer-content");

    viewerContent.addEventListener("pointerdown", (e) => {
        if (e.target.closest(".viewer-close") || e.target.closest(".viewer-meta")) return;
        pointers.set(e.pointerId, e);
        viewerContent.setPointerCapture(e.pointerId);

        if (pointers.size === 2) {
            const pts = Array.from(pointers.values());
            lastPinchDist = getDistance(pts[0], pts[1]);
            lastPinchCenter = getCenter(pts[0], pts[1]);
        }

        // Double-tap detection
        if (pointers.size === 1) {
            const now = Date.now();
            if (now - lastTapTime < 300) {
                // Toggle 1x ↔ 2x
                if (viewerState.scale > 1.1) {
                    viewerState.scale = 1;
                    viewerState.x = 0;
                    viewerState.y = 0;
                } else {
                    const rect = viewerContent.getBoundingClientRect();
                    const cx = e.clientX - rect.left - rect.width / 2;
                    const cy = e.clientY - rect.top - rect.height / 2;
                    viewerState.scale = 2;
                    viewerState.x = -cx;
                    viewerState.y = -cy;
                }
                applyViewerTransform();
                lastTapTime = 0;
            } else {
                lastTapTime = now;
            }
        }
    });

    viewerContent.addEventListener("pointermove", (e) => {
        if (!pointers.has(e.pointerId)) return;
        const prev = pointers.get(e.pointerId);
        pointers.set(e.pointerId, e);

        if (pointers.size === 1 && viewerState.scale > 1) {
            // Pan
            viewerState.x += e.clientX - prev.clientX;
            viewerState.y += e.clientY - prev.clientY;
            applyViewerTransform();
        } else if (pointers.size === 2) {
            // Pinch zoom
            const pts = Array.from(pointers.values());
            const dist = getDistance(pts[0], pts[1]);
            const center = getCenter(pts[0], pts[1]);

            if (lastPinchDist > 0) {
                const ratio = dist / lastPinchDist;
                const newScale = Math.min(5, Math.max(1, viewerState.scale * ratio));

                if (lastPinchCenter) {
                    const rect = viewerContent.getBoundingClientRect();
                    const cx = center.x - rect.left;
                    const cy = center.y - rect.top;
                    const ds = newScale / viewerState.scale;
                    viewerState.x = cx - ds * (cx - viewerState.x);
                    viewerState.y = cy - ds * (cy - viewerState.y);
                }

                viewerState.scale = newScale;
                applyViewerTransform();
            }

            lastPinchDist = dist;
            lastPinchCenter = center;
        }
    });

    viewerContent.addEventListener("pointerup", (e) => {
        pointers.delete(e.pointerId);
        if (pointers.size < 2) {
            lastPinchDist = 0;
            lastPinchCenter = null;
        }
    });

    viewerContent.addEventListener("pointercancel", (e) => {
        pointers.delete(e.pointerId);
        if (pointers.size < 2) {
            lastPinchDist = 0;
            lastPinchCenter = null;
        }
    });

    // Mouse wheel zoom (desktop)
    viewerContent.addEventListener("wheel", (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.min(5, Math.max(1, viewerState.scale * delta));

        const rect = viewerContent.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const ds = newScale / viewerState.scale;
        viewerState.x = cx - ds * (cx - viewerState.x);
        viewerState.y = cy - ds * (cy - viewerState.y);

        viewerState.scale = newScale;
        if (viewerState.scale <= 1) { viewerState.x = 0; viewerState.y = 0; }
        applyViewerTransform();
    }, { passive: false });

    // ── Highlight overlay ─────────────────────────────────────
    window.drawHighlight = function (img) {
        const canvas = img.parentElement.querySelector("canvas");
        if (!canvas) return;

        const tileX = +img.dataset.tileX;
        const tileY = +img.dataset.tileY;
        const tileW = +img.dataset.tileW;
        const tileH = +img.dataset.tileH;

        // Natural (actual) image dimensions
        const natW = img.naturalWidth;
        const natH = img.naturalHeight;

        // Full-page match — no highlight needed
        if (tileX === 0 && tileY === 0 && tileW === natW && tileH === natH) return;

        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;

        const scaleX = canvas.width / natW;
        const scaleY = canvas.height / natH;

        const ctx = canvas.getContext("2d");
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 3;
        ctx.strokeRect(
            tileX * scaleX,
            tileY * scaleY,
            tileW * scaleX,
            tileH * scaleY
        );
    };

    // ── Catalogs ──────────────────────────────────────────────
    async function loadCatalogs() {
        catalogsLoading.classList.remove("hidden");
        catalogsList.innerHTML = "";
        catalogsEmpty.classList.add("hidden");

        try {
            const res = await fetch("/api/catalogs");
            if (!res.ok) throw new Error("Failed to load catalogs");
            const data = await res.json();

            if (!data.catalogs.length) {
                catalogsEmpty.classList.remove("hidden");
                return;
            }

            catalogsList.innerHTML = data.catalogs
                .map(
                    (c) =>
                        '<div class="catalog-card">' +
                        '  <div class="catalog-info">' +
                        "    <h3>" + escapeHtml(c.pdf_name) + "</h3>" +
                        "    <p>" + c.num_pages + " pages &middot; " + c.num_vectors + " vectors</p>" +
                        "  </div>" +

                        "</div>"
                )
                .join("");
        } catch (err) {
            catalogsList.innerHTML =
                '<p class="empty-state">Error loading catalogs.</p>';
        } finally {
            catalogsLoading.classList.add("hidden");
        }
    }

    // ── Pages (tile viewer) ───────────────────────────────────
    const pagesCatalogSelect = document.getElementById("pages-catalog-select");
    const pagesNav = document.getElementById("pages-nav");
    const pagesPrev = document.getElementById("pages-prev");
    const pagesNext = document.getElementById("pages-next");
    const pagesIndicator = document.getElementById("pages-indicator");
    const pagesLoading = document.getElementById("pages-loading");
    const pagesViewer = document.getElementById("pages-viewer");
    const pagesImage = document.getElementById("pages-image");
    const pagesCanvas = document.getElementById("pages-canvas");
    const pagesImageWrap = document.getElementById("pages-image-wrap");
    const pagesTileCount = document.getElementById("pages-tile-count");
    const pagesEmpty = document.getElementById("pages-empty");

    let pagesData = [];      // [{page_number, page_image}, ...]
    let pagesIndex = 0;      // current index into pagesData
    let pagesTiles = [];     // current page tiles [{tile_x, tile_y, tile_w, tile_h}]

    // Tile overlay colors (cycling)
    const TILE_COLORS = [
        { stroke: "#3b82f6", fill: "rgba(59,130,246,0.12)" },
        { stroke: "#22c55e", fill: "rgba(34,197,94,0.12)" },
        { stroke: "#eab308", fill: "rgba(234,179,8,0.12)" },
        { stroke: "#ef4444", fill: "rgba(239,68,68,0.12)" },
        { stroke: "#a855f7", fill: "rgba(168,85,247,0.12)" },
        { stroke: "#06b6d4", fill: "rgba(6,182,212,0.12)" },
    ];

    async function loadPagesCatalogList() {
        try {
            const res = await fetch("/api/catalogs");
            if (!res.ok) return;
            const data = await res.json();

            pagesCatalogSelect.innerHTML = '<option value="">Select a catalog...</option>' +
                data.catalogs.map(function (c) {
                    return '<option value="' + escapeHtml(c.pdf_name) + '">' +
                        escapeHtml(c.pdf_name) + " (" + c.num_pages + " pages)</option>";
                }).join("");
        } catch (_) {}
    }

    pagesCatalogSelect.addEventListener("change", async function () {
        var catalog = pagesCatalogSelect.value;
        pagesNav.classList.add("hidden");
        pagesViewer.classList.add("hidden");
        pagesEmpty.classList.add("hidden");
        pagesData = [];
        pagesIndex = 0;

        if (!catalog) {
            pagesEmpty.classList.remove("hidden");
            return;
        }

        pagesLoading.classList.remove("hidden");
        try {
            var res = await fetch("/api/pages?catalog=" + encodeURIComponent(catalog));
            if (!res.ok) throw new Error("Failed to load pages");
            var data = await res.json();
            pagesData = data.pages;
            pagesIndex = 0;
            pagesNav.classList.remove("hidden");
            updatePagesNav();
            await loadPageTiles();
        } catch (err) {
            pagesViewer.innerHTML = '<p class="empty-state">Error: ' + escapeHtml(err.message) + "</p>";
            pagesViewer.classList.remove("hidden");
        } finally {
            pagesLoading.classList.add("hidden");
        }
    });

    pagesPrev.addEventListener("click", async function () {
        if (pagesIndex > 0) {
            pagesIndex--;
            updatePagesNav();
            await loadPageTiles();
        }
    });

    pagesNext.addEventListener("click", async function () {
        if (pagesIndex < pagesData.length - 1) {
            pagesIndex++;
            updatePagesNav();
            await loadPageTiles();
        }
    });

    function updatePagesNav() {
        pagesIndicator.textContent = (pagesIndex + 1) + " / " + pagesData.length;
        pagesPrev.disabled = pagesIndex <= 0;
        pagesNext.disabled = pagesIndex >= pagesData.length - 1;
    }

    async function loadPageTiles() {
        var p = pagesData[pagesIndex];
        if (!p) return;

        pagesLoading.classList.remove("hidden");
        pagesViewer.classList.add("hidden");
        try {
            var catalog = pagesCatalogSelect.value;
            var res = await fetch("/api/pages?catalog=" + encodeURIComponent(catalog) +
                "&page=" + p.page_number);
            if (!res.ok) throw new Error("Failed to load tiles");
            var data = await res.json();

            pagesTiles = data.tiles;
            pagesTileCount.textContent = data.total_tiles + " tile" + (data.total_tiles !== 1 ? "s" : "") +
                " on page " + p.page_number;

            pagesImage.src = data.page_image;
            pagesImage.onload = function () {
                pagesViewer.classList.remove("hidden");
                drawPageTiles();
                // Reset zoom
                pagesViewerState.scale = 1;
                pagesViewerState.x = 0;
                pagesViewerState.y = 0;
                applyPagesTransform();
            };
        } catch (err) {
            pagesTileCount.textContent = "Error: " + err.message;
            pagesViewer.classList.remove("hidden");
        } finally {
            pagesLoading.classList.add("hidden");
        }
    }

    function drawPageTiles() {
        var natW = pagesImage.naturalWidth;
        var natH = pagesImage.naturalHeight;
        var dispW = pagesImage.clientWidth;
        var dispH = pagesImage.clientHeight;

        pagesCanvas.width = dispW;
        pagesCanvas.height = dispH;

        var scaleX = dispW / natW;
        var scaleY = dispH / natH;

        var ctx = pagesCanvas.getContext("2d");
        ctx.clearRect(0, 0, dispW, dispH);

        pagesTiles.forEach(function (tile, i) {
            var color = TILE_COLORS[i % TILE_COLORS.length];
            var x = tile.tile_x * scaleX;
            var y = tile.tile_y * scaleY;
            var w = tile.tile_w * scaleX;
            var h = tile.tile_h * scaleY;

            // Semi-transparent fill
            ctx.fillStyle = color.fill;
            ctx.fillRect(x, y, w, h);

            // Dashed border
            ctx.strokeStyle = color.stroke;
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);

            // Tile index label
            ctx.fillStyle = color.stroke;
            ctx.font = "bold 11px -apple-system, sans-serif";
            var label = String(i + 1);
            var textW = ctx.measureText(label).width;
            var pad = 3;
            ctx.fillStyle = color.stroke;
            ctx.globalAlpha = 0.85;
            ctx.fillRect(x, y, textW + pad * 2, 16);
            ctx.globalAlpha = 1;
            ctx.fillStyle = "#fff";
            ctx.fillText(label, x + pad, y + 12);
        });
    }

    // Redraw tiles on window resize
    window.addEventListener("resize", function () {
        if (!pagesViewer.classList.contains("hidden") && pagesTiles.length) {
            drawPageTiles();
        }
    });

    // ── Pages viewer pan & zoom ───────────────────────────────
    var pagesViewerState = { scale: 1, x: 0, y: 0 };
    var pagesPointers = new Map();
    var pagesLastPinchDist = 0;
    var pagesLastPinchCenter = null;
    var pagesLastTap = 0;

    function applyPagesTransform() {
        pagesImageWrap.style.transform =
            "translate(" + pagesViewerState.x + "px," + pagesViewerState.y + "px) scale(" + pagesViewerState.scale + ")";
    }

    pagesImageWrap.addEventListener("pointerdown", function (e) {
        pagesPointers.set(e.pointerId, e);
        pagesImageWrap.setPointerCapture(e.pointerId);

        if (pagesPointers.size === 2) {
            var pts = Array.from(pagesPointers.values());
            pagesLastPinchDist = getDistance(pts[0], pts[1]);
            pagesLastPinchCenter = getCenter(pts[0], pts[1]);
        }

        if (pagesPointers.size === 1) {
            var now = Date.now();
            if (now - pagesLastTap < 300) {
                if (pagesViewerState.scale > 1.1) {
                    pagesViewerState.scale = 1;
                    pagesViewerState.x = 0;
                    pagesViewerState.y = 0;
                } else {
                    pagesViewerState.scale = 2;
                }
                applyPagesTransform();
                pagesLastTap = 0;
            } else {
                pagesLastTap = now;
            }
        }
    });

    pagesImageWrap.addEventListener("pointermove", function (e) {
        if (!pagesPointers.has(e.pointerId)) return;
        var prev = pagesPointers.get(e.pointerId);
        pagesPointers.set(e.pointerId, e);

        if (pagesPointers.size === 1 && pagesViewerState.scale > 1) {
            pagesViewerState.x += e.clientX - prev.clientX;
            pagesViewerState.y += e.clientY - prev.clientY;
            applyPagesTransform();
        } else if (pagesPointers.size === 2) {
            var pts = Array.from(pagesPointers.values());
            var dist = getDistance(pts[0], pts[1]);
            if (pagesLastPinchDist > 0) {
                var ratio = dist / pagesLastPinchDist;
                pagesViewerState.scale = Math.min(5, Math.max(1, pagesViewerState.scale * ratio));
                applyPagesTransform();
            }
            pagesLastPinchDist = dist;
        }
    });

    pagesImageWrap.addEventListener("pointerup", function (e) {
        pagesPointers.delete(e.pointerId);
        if (pagesPointers.size < 2) { pagesLastPinchDist = 0; pagesLastPinchCenter = null; }
    });

    pagesImageWrap.addEventListener("pointercancel", function (e) {
        pagesPointers.delete(e.pointerId);
        if (pagesPointers.size < 2) { pagesLastPinchDist = 0; pagesLastPinchCenter = null; }
    });

    pagesImageWrap.addEventListener("wheel", function (e) {
        e.preventDefault();
        var delta = e.deltaY > 0 ? 0.9 : 1.1;
        pagesViewerState.scale = Math.min(5, Math.max(1, pagesViewerState.scale * delta));
        if (pagesViewerState.scale <= 1) { pagesViewerState.x = 0; pagesViewerState.y = 0; }
        applyPagesTransform();
    }, { passive: false });

    // ── Utilities ─────────────────────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }
})();
