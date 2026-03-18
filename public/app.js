/* fab-search — Frontend application logic */

(function () {
    "use strict";

    // ── DOM refs ──────────────────────────────────────────────
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const clearBtn = document.getElementById("clear-btn");
    const topKSlider = document.getElementById("top-k");
    const topKValue = document.getElementById("top-k-value");
    const searchBtn = document.getElementById("search-btn");
    const loading = document.getElementById("loading");
    const resultsContainer = document.getElementById("results");

    const catalogsLoading = document.getElementById("catalogs-loading");
    const catalogsList = document.getElementById("catalogs-list");
    const catalogsEmpty = document.getElementById("catalogs-empty");

    let selectedFile = null;

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
        });
    });

    // ── File upload ───────────────────────────────────────────
    uploadArea.addEventListener("click", () => fileInput.click());

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

    function handleFile(file) {
        if (!file.type.startsWith("image/")) return;
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.classList.remove("hidden");
            uploadArea.classList.add("hidden");
            searchBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        previewContainer.classList.add("hidden");
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
            const res = await fetch("/api/search?top_k=" + encodeURIComponent(topK), {
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
                        '  <button class="btn-danger" onclick="removeCatalog(\'' +
                        escapeHtml(c.pdf_name).replace(/'/g, "\\'") +
                        "')\">" +
                        "    Remove" +
                        "  </button>" +
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

    window.removeCatalog = async function (name) {
        if (!confirm("Remove catalog \"" + name + "\"? This cannot be undone.")) return;

        try {
            const res = await fetch("/api/catalogs/" + encodeURIComponent(name), {
                method: "DELETE",
            });
            if (!res.ok) throw new Error("Delete failed");
            loadCatalogs();
        } catch (err) {
            alert("Failed to remove catalog: " + err.message);
        }
    };

    // ── Utilities ─────────────────────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }
})();
