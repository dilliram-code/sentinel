/**
 * Campus Sentinel — Modern Web Application Controller
 * Handles WebSocket binary streaming, real-time telemetry HUD,
 * Chart.js analytics, REST API integrations, and modal interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
    // -----------------------------------------------------------------------
    // State Management
    // -----------------------------------------------------------------------
    const state = {
        activeTab: 'live',
        ws: null,
        wsConnected: false,
        pipelineRunning: false,
        lastBlobUrl: null,
        charts: {},
        stats: {
            stakeholders: 0,
            visits: 0,
            visits_today: 0,
            unknowns: 0,
            unverified_unknowns: 0
        },
        videos: []
    };

    // -----------------------------------------------------------------------
    // DOM Selectors
    // -----------------------------------------------------------------------
    const dom = {
        // Nav & Tabs
        navBtns: document.querySelectorAll('.nav-btn'),
        tabPanes: document.querySelectorAll('.tab-pane'),
        pageTitle: document.getElementById('pageTitle'),
        pageSubtitle: document.getElementById('pageSubtitle'),

        // Status & Clock
        liveClock: document.getElementById('liveClock'),
        pipelineStatusText: document.getElementById('pipelineStatusText'),
        aiDeviceText: document.getElementById('aiDeviceText'),
        statusDot: document.getElementById('statusDot'),
        statusPulse: document.getElementById('statusPulse'),

        // Headline Metrics
        statStakeholders: document.getElementById('statStakeholders'),
        statVisitsToday: document.getElementById('statVisitsToday'),
        statTotalVisits: document.getElementById('statTotalVisits'),
        statUnknowns: document.getElementById('statUnknowns'),
        statUnverifiedBadge: document.getElementById('statUnverifiedBadge'),
        navVisitCount: document.getElementById('navVisitCount'),
        navUnknownCount: document.getElementById('navUnknownCount'),
        navStakeholderCount: document.getElementById('navStakeholderCount'),

        // Pipeline Actions
        btnStartPipeline: document.getElementById('btnStartPipeline'),
        btnStopPipeline: document.getElementById('btnStopPipeline'),
        btnQuickConfig: document.getElementById('btnQuickConfig'),

        // Live Feed Elements
        liveVideoCanvas: document.getElementById('liveVideoCanvas'),
        liveVideoFeed: document.getElementById('liveVideoFeed'),
        streamOverlay: document.getElementById('streamOverlay'),
        streamStatusText: document.getElementById('streamStatusText'),
        btnOverlayStart: document.getElementById('btnOverlayStart'),
        hudFps: document.getElementById('hudFps'),
        hudPersons: document.getElementById('hudPersons'),
        hudRecognized: document.getElementById('hudRecognized'),
        hudUnknown: document.getElementById('hudUnknown'),
        currentCameraLocation: document.getElementById('currentCameraLocation'),
        btnSnapshot: document.getElementById('btnSnapshot'),
        btnFullscreen: document.getElementById('btnFullscreen'),
        videoContainer: document.getElementById('videoContainer'),
        liveEventsList: document.getElementById('liveEventsList'),

        // Visit Logs Tab
        visitSearchInput: document.getElementById('visitSearchInput'),
        visitRoleFilter: document.getElementById('visitRoleFilter'),
        btnExportVisits: document.getElementById('btnExportVisits'),
        visitsTableBody: document.getElementById('visitsTableBody'),

        // Unknowns Tab
        chkOnlyUnverified: document.getElementById('chkOnlyUnverified'),
        btnRefreshUnknowns: document.getElementById('btnRefreshUnknowns'),
        unknownsGrid: document.getElementById('unknownsGrid'),

        // Stakeholders Tab
        stakeholdersGrid: document.getElementById('stakeholdersGrid'),
        btnOpenRegisterModal: document.getElementById('btnOpenRegisterModal'),

        // Modals
        modalRegister: document.getElementById('modalRegister'),
        btnCloseRegisterModal: document.getElementById('btnCloseRegisterModal'),
        btnCancelRegister: document.getElementById('btnCancelRegister'),
        formRegisterStakeholder: document.getElementById('formRegisterStakeholder'),
        photoDropzone: document.getElementById('photoDropzone'),
        photoDropzoneText: document.getElementById('photoDropzoneText'),
        regPhotoInput: document.getElementById('regPhotoInput'),
        photoPreviewContainer: document.getElementById('photoPreviewContainer'),
        photoPreviewImg: document.getElementById('photoPreviewImg'),

        // Settings Tab
        inputSourceModes: document.querySelectorAll('input[name="inputSourceMode"]'),
        webcamConfigBox: document.getElementById('webcamConfigBox'),
        rtspConfigBox: document.getElementById('rtspConfigBox'),
        videoConfigBox: document.getElementById('videoConfigBox'),
        cfgWebcamIndex: document.getElementById('cfgWebcamIndex'),
        cfgRtspUrl: document.getElementById('cfgRtspUrl'),
        cfgVideoSelect: document.getElementById('cfgVideoSelect'),
        cfgLocationLabel: document.getElementById('cfgLocationLabel'),
        btnSaveApplyConfig: document.getElementById('btnSaveApplyConfig'),
        videoDropzone: document.getElementById('videoDropzone'),
        videoFileInput: document.getElementById('videoFileInput'),

        // Toast Container
        toastContainer: document.getElementById('toastContainer')
    };

    // -----------------------------------------------------------------------
    // Live Clock
    // -----------------------------------------------------------------------
    function updateClock() {
        const now = new Date();
        dom.liveClock.textContent = now.toLocaleTimeString();
    }
    setInterval(updateClock, 1000);
    updateClock();

    // -----------------------------------------------------------------------
    // Toast Notifications
    // -----------------------------------------------------------------------
    function showToast(title, message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        let iconName = 'info';
        if (type === 'visit') iconName = 'check-circle-2';
        if (type === 'unknown') iconName = 'alert-triangle';

        toast.innerHTML = `
            <i data-lucide="${iconName}" class="icon-sm"></i>
            <div>
                <strong style="display:block; font-size: 0.88rem;">${title}</strong>
                <span style="font-size: 0.78rem; opacity: 0.85;">${message}</span>
            </div>
        `;

        dom.toastContainer.appendChild(toast);
        if (window.lucide) lucide.createIcons();

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // -----------------------------------------------------------------------
    // WebSocket High-Performance Video Receiver
    // -----------------------------------------------------------------------
    let canvasCtx = null;
    if (dom.liveVideoCanvas) {
        canvasCtx = dom.liveVideoCanvas.getContext('2d', { alpha: false });
    }

    function renderImgFallback(blob) {
        const newUrl = URL.createObjectURL(blob);
        dom.liveVideoFeed.onload = () => {
            if (state.lastBlobUrl && state.lastBlobUrl !== newUrl) {
                URL.revokeObjectURL(state.lastBlobUrl);
            }
            state.lastBlobUrl = newUrl;
        };
        dom.liveVideoFeed.src = newUrl;
        dom.streamOverlay.classList.add('hidden');
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/live`;

        state.ws = new WebSocket(wsUrl);
        state.ws.binaryType = 'arraybuffer';

        state.ws.onopen = () => {
            state.wsConnected = true;
            log('Live WebSocket connection established.');
        };

        state.ws.onmessage = (event) => {
            // Binary Message: Video Frame
            if (event.data instanceof ArrayBuffer) {
                const blob = new Blob([event.data], { type: 'image/jpeg' });

                // Fast GPU Canvas rendering via createImageBitmap (Zero GC churn, 60fps capable)
                if (window.createImageBitmap && dom.liveVideoCanvas && canvasCtx) {
                    createImageBitmap(blob).then((bitmap) => {
                        if (dom.liveVideoCanvas.width !== bitmap.width || dom.liveVideoCanvas.height !== bitmap.height) {
                            dom.liveVideoCanvas.width = bitmap.width;
                            dom.liveVideoCanvas.height = bitmap.height;
                        }
                        if (dom.liveVideoCanvas.style.display !== 'block') {
                            dom.liveVideoCanvas.style.display = 'block';
                            dom.liveVideoFeed.style.display = 'none';
                        }
                        canvasCtx.drawImage(bitmap, 0, 0);
                        bitmap.close();
                        dom.streamOverlay.classList.add('hidden');
                    }).catch(() => {
                        renderImgFallback(blob);
                    });
                } else {
                    renderImgFallback(blob);
                }
            } 
            // JSON Message: Telemetry or Events
            else {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'events' && Array.isArray(msg.data)) {
                        msg.data.forEach(evt => handleLiveEvent(evt));
                    }
                    if (msg.type === 'status') {
                        updatePipelineUI(msg.data);
                    }
                } catch (e) {
                    console.error('Error parsing WS JSON message:', e);
                }
            }
        };

        state.ws.onclose = () => {
            state.wsConnected = false;
            dom.streamOverlay.classList.remove('hidden');
            dom.streamStatusText.textContent = 'Reconnecting to stream server...';
            setTimeout(connectWebSocket, 2000);
        };

        state.ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    }

    function handleLiveEvent(evt) {
        const timeStr = new Date(evt.timestamp * 1000).toLocaleTimeString();
        const item = document.createElement('div');

        if (evt.type === 'STAKEHOLDER_VISIT') {
            item.className = 'event-item event-visit';
            item.innerHTML = `
                <div class="event-details">
                    <span class="event-title">🎓 ${evt.name}</span>
                    <span class="event-meta">${evt.role} • ${evt.location} (Match: ${(evt.similarity * 100).toFixed(1)}%)</span>
                </div>
                <span class="event-time">${timeStr}</span>
            `;
            showToast('Stakeholder Identified', `${evt.name} (${evt.role}) at ${evt.location}`, 'visit');
        } else if (evt.type === 'UNKNOWN_CAPTURED') {
            item.className = 'event-item event-unknown';
            item.innerHTML = `
                <div class="event-details">
                    <span class="event-title text-rose">🚨 Unknown Individual</span>
                    <span class="event-meta">Logged at ${evt.location}</span>
                </div>
                <span class="event-time">${timeStr}</span>
            `;
            showToast('Unknown Person Detected', `Captured at ${evt.location}`, 'unknown');
        }

        // Prepend to event list
        const emptyState = dom.liveEventsList.querySelector('.event-empty-state');
        if (emptyState) emptyState.remove();

        dom.liveEventsList.insertBefore(item, dom.liveEventsList.firstChild);

        // Keep list bounded to 30 items
        while (dom.liveEventsList.children.length > 30) {
            dom.liveEventsList.lastChild.remove();
        }

        // Refresh stats on event
        fetchStats();
    }

    // -----------------------------------------------------------------------
    // REST API Integration & Data Fetchers
    // -----------------------------------------------------------------------
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const data = await res.json();
            state.stats = data;

            dom.statStakeholders.textContent = data.stakeholders || 0;
            dom.statVisitsToday.textContent = data.visits_today || 0;
            dom.statTotalVisits.textContent = data.visits || 0;
            dom.statUnknowns.textContent = data.unknowns || 0;

            dom.navVisitCount.textContent = data.visits_today || 0;
            dom.navUnknownCount.textContent = data.unverified_unknowns || 0;
            dom.navStakeholderCount.textContent = data.stakeholders || 0;

            if (data.pipeline) {
                updatePipelineUI(data.pipeline);
            }
        } catch (err) {
            console.error('Failed to fetch stats:', err);
        }
    }

    function updatePipelineUI(pipeline) {
        state.pipelineRunning = pipeline.running;

        if (pipeline.running) {
            dom.pipelineStatusText.textContent = `Pipeline Active (${pipeline.fps || 0} FPS)`;
            dom.statusDot.style.background = 'var(--emerald)';
            dom.statusPulse.style.borderColor = 'var(--emerald)';
            dom.btnStartPipeline.style.display = 'none';
            dom.btnStopPipeline.style.display = 'inline-flex';
            dom.streamOverlay.classList.add('hidden');

            dom.hudFps.textContent = pipeline.fps || 0.0;
            if (pipeline.summary) {
                dom.hudPersons.textContent = pipeline.summary.persons || 0;
                dom.hudRecognized.textContent = pipeline.summary.recognized || 0;
                dom.hudUnknown.textContent = pipeline.summary.unknown || 0;
            }
            dom.currentCameraLocation.textContent = pipeline.location || 'Surveillance Feed';
        } else {
            dom.pipelineStatusText.textContent = 'Pipeline Idle';
            dom.statusDot.style.background = 'var(--rose)';
            dom.statusPulse.style.borderColor = 'var(--rose)';
            dom.btnStartPipeline.style.display = 'inline-flex';
            dom.btnStopPipeline.style.display = 'none';
            dom.streamOverlay.classList.remove('hidden');
            dom.streamStatusText.textContent = 'Camera stream is stopped. Click Start Camera to begin.';
        }

        if (pipeline.ai_device) {
            dom.aiDeviceText.textContent = `Device: ${pipeline.ai_device.toUpperCase()}`;
        }
    }

    // Start / Stop Pipeline
    async function startPipeline(source = null, location = null) {
        const formData = new FormData();
        if (source !== null) formData.append('source', source);
        if (location) formData.append('location', location);

        try {
            const res = await fetch('/api/pipeline/start', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (res.ok) {
                showToast('Pipeline Started', 'Real-time AI monitoring active.', 'visit');
                fetchStats();
            } else {
                showToast('Start Failed', data.detail || 'Could not start stream.', 'unknown');
            }
        } catch (err) {
            showToast('Network Error', err.message, 'unknown');
        }
    }

    async function stopPipeline() {
        try {
            const res = await fetch('/api/pipeline/stop', { method: 'POST' });
            const data = await res.json();
            if (res.ok) {
                showToast('Pipeline Stopped', 'Camera feed released.', 'info');
                fetchStats();
            }
        } catch (err) {
            console.error('Stop failed:', err);
        }
    }

    // -----------------------------------------------------------------------
    // Tab Data Loaders
    // -----------------------------------------------------------------------
    async function loadVisitsTab() {
        const search = dom.visitSearchInput.value.trim();
        const role = dom.visitRoleFilter.value;
        const params = new URLSearchParams({ limit: 200, offset: 0 });
        if (search) params.append('search', search);
        if (role) params.append('role', role);

        try {
            const res = await fetch(`/api/visits?${params.toString()}`);
            const data = await res.json();

            if (!data.visits || data.visits.length === 0) {
                dom.visitsTableBody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center py-4" style="color: var(--text-muted);">
                            No visit records found matching criteria.
                        </td>
                    </tr>
                `;
                return;
            }

            dom.visitsTableBody.innerHTML = data.visits.map(v => {
                const roleClass = `role-${v.role.toLowerCase()}`;
                const avatar = v.image_path || '/static/assets/placeholder.png';
                const simPercent = (v.similarity * 100).toFixed(1);

                return `
                    <tr>
                        <td>
                            <div class="user-cell">
                                <img class="user-avatar" src="${avatar}" onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(v.name)}&background=6366f1&color=fff'" alt="" />
                                <div class="user-meta">
                                    <span class="user-name">${v.name}</span>
                                    <span class="user-uid">${v.uid}</span>
                                </div>
                            </div>
                        </td>
                        <td><span class="role-badge ${roleClass}">${v.role}</span></td>
                        <td><code>${v.uid}</code></td>
                        <td>${v.location}</td>
                        <td>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; width: 60px;">
                                    <div style="width: ${simPercent}%; height: 100%; background: var(--emerald);"></div>
                                </div>
                                <span style="font-size: 0.78rem; font-weight: 600; color: var(--emerald);">${simPercent}%</span>
                            </div>
                        </td>
                        <td style="color: var(--text-muted); font-size: 0.8rem;">${v.timestamp.replace('T', ' ')}</td>
                    </tr>
                `;
            }).join('');
        } catch (err) {
            console.error('Failed to load visits:', err);
        }
    }

    async function loadUnknownsTab() {
        const onlyUnverified = dom.chkOnlyUnverified.checked;
        try {
            const res = await fetch(`/api/unknowns?limit=100&only_unverified=${onlyUnverified}`);
            const data = await res.json();

            if (!data.unknowns || data.unknowns.length === 0) {
                dom.unknownsGrid.innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">
                        <i data-lucide="check-circle" style="width: 48px; height: 48px; color: var(--emerald); margin-bottom: 12px;"></i>
                        <h3>All Clear! No unknown persons in queue.</h3>
                    </div>
                `;
                if (window.lucide) lucide.createIcons();
                return;
            }

            dom.unknownsGrid.innerHTML = data.unknowns.map(u => {
                const img = u.image_url || '/static/assets/placeholder.png';
                const statusBadge = u.verified 
                    ? `<span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7;">✔ Verified</span>`
                    : `<span class="badge badge-rose">Pending</span>`;

                return `
                    <div class="person-card">
                        <div class="person-card-img-wrapper">
                            <img class="person-card-img" src="${img}" alt="Unknown face" onerror="this.style.opacity=0.3" />
                            <div style="position: absolute; top: 10px; right: 10px;">${statusBadge}</div>
                        </div>
                        <div class="person-card-body">
                            <span class="person-card-title">Unknown #${u.id}</span>
                            <span class="person-card-meta"><i data-lucide="map-pin" class="icon-sm"></i> ${u.location}</span>
                            <span class="person-card-meta"><i data-lucide="clock" class="icon-sm"></i> ${u.timestamp.replace('T', ' ')}</span>
                            <div class="person-card-actions">
                                ${!u.verified ? `
                                    <button class="btn btn-primary btn-sm" onclick="window.verifyUnknown(${u.id})">
                                        <i data-lucide="check" class="icon-sm"></i> Verify
                                    </button>
                                ` : ''}
                                <button class="btn btn-secondary btn-sm" onclick="window.deleteUnknown(${u.id})">
                                    <i data-lucide="trash-2" class="icon-sm"></i> Delete
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');

            if (window.lucide) lucide.createIcons();
        } catch (err) {
            console.error('Failed to load unknowns:', err);
        }
    }

    async function loadStakeholdersTab() {
        try {
            const res = await fetch('/api/stakeholders');
            const data = await res.json();

            if (!data.stakeholders || data.stakeholders.length === 0) {
                dom.stakeholdersGrid.innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">
                        <h3>No stakeholders enrolled yet. Click 'Register New Stakeholder' to add people.</h3>
                    </div>
                `;
                return;
            }

            dom.stakeholdersGrid.innerHTML = data.stakeholders.map(s => {
                const roleClass = `role-${s.role.toLowerCase()}`;
                const avatar = s.image_url || `https://ui-avatars.com/api/?name=${encodeURIComponent(s.name)}&background=6366f1&color=fff`;

                return `
                    <div class="person-card">
                        <div class="person-card-img-wrapper">
                            <img class="person-card-img" src="${avatar}" alt="${s.name}" />
                            <div style="position: absolute; top: 10px; right: 10px;">
                                <span class="role-badge ${roleClass}">${s.role}</span>
                            </div>
                        </div>
                        <div class="person-card-body">
                            <span class="person-card-title">${s.name}</span>
                            <span class="person-card-meta"><b>UID:</b> <code>${s.uid}</code></span>
                            <span class="person-card-meta"><i data-lucide="calendar" class="icon-sm"></i> Registered: ${s.registered_at ? s.registered_at.substring(0, 10) : 'N/A'}</span>
                            <div class="person-card-actions">
                                <button class="btn btn-rose btn-sm" onclick="window.deleteStakeholder('${s.uid}', '${s.name}')">
                                    <i data-lucide="trash" class="icon-sm"></i> Remove
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');

            if (window.lucide) lucide.createIcons();
        } catch (err) {
            console.error('Failed to load stakeholders:', err);
        }
    }

    async function loadAnalyticsTab() {
        try {
            const res = await fetch('/api/reports');
            const data = await res.json();

            // Destroy previous instances if any
            if (state.charts.daily) state.charts.daily.destroy();
            if (state.charts.roles) state.charts.roles.destroy();
            if (state.charts.locations) state.charts.locations.destroy();

            // 1. Daily Visits Chart (Area gradient)
            const ctxDaily = document.getElementById('chartDailyVisits').getContext('2d');
            const gradientDaily = ctxDaily.createLinearGradient(0, 0, 0, 300);
            gradientDaily.addColorStop(0, 'rgba(99, 102, 241, 0.5)');
            gradientDaily.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

            state.charts.daily = new Chart(ctxDaily, {
                type: 'line',
                data: {
                    labels: data.daily_visits.map(d => d.date),
                    datasets: [{
                        label: 'Campus Visits',
                        data: data.daily_visits.map(d => d.count),
                        borderColor: '#6366f1',
                        backgroundColor: gradientDaily,
                        fill: true,
                        tension: 0.35,
                        pointRadius: 4,
                        pointBackgroundColor: '#818cf8'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', precision: 0 } }
                    }
                }
            });

            // 2. Roles Doughnut Chart
            const ctxRole = document.getElementById('chartRoleVisits').getContext('2d');
            state.charts.roles = new Chart(ctxRole, {
                type: 'doughnut',
                data: {
                    labels: data.role_visits.map(r => r.role),
                    datasets: [{
                        data: data.role_visits.map(r => r.count),
                        backgroundColor: ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#e2e8f0', font: { family: 'Inter' } } }
                    }
                }
            });

            // 3. Location Bar Chart
            const ctxLoc = document.getElementById('chartLocationVisits').getContext('2d');
            state.charts.locations = new Chart(ctxLoc, {
                type: 'bar',
                data: {
                    labels: data.cam_visits.map(l => l.location),
                    datasets: [{
                        label: 'Check-ins',
                        data: data.cam_visits.map(l => l.count),
                        backgroundColor: '#06b6d4',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', precision: 0 } }
                    }
                }
            });

        } catch (err) {
            console.error('Failed to load reports:', err);
        }
    }

    async function loadSettingsTab() {
        try {
            const res = await fetch('/api/videos');
            const data = await res.json();
            state.videos = data.videos || [];

            dom.cfgVideoSelect.innerHTML = '<option value="">Select a video...</option>' + 
                state.videos.map(v => `<option value="${v}">${v}</option>`).join('');
        } catch (err) {
            console.error('Failed to load videos list:', err);
        }
    }

    // -----------------------------------------------------------------------
    // Global Window Callbacks (for button actions)
    // -----------------------------------------------------------------------
    window.verifyUnknown = async (id) => {
        try {
            const res = await fetch(`/api/unknowns/${id}/verify`, { method: 'POST' });
            if (res.ok) {
                showToast('Verified', `Unknown individual #${id} verified.`, 'visit');
                loadUnknownsTab();
                fetchStats();
            }
        } catch (err) {
            console.error(err);
        }
    };

    window.deleteUnknown = async (id) => {
        if (!confirm('Are you sure you want to delete this capture record?')) return;
        try {
            const res = await fetch(`/api/unknowns/${id}`, { method: 'DELETE' });
            if (res.ok) {
                showToast('Deleted', 'Capture removed.', 'info');
                loadUnknownsTab();
                fetchStats();
            }
        } catch (err) {
            console.error(err);
        }
    };

    window.deleteStakeholder = async (uid, name) => {
        if (!confirm(`Are you sure you want to delete ${name} (${uid})? This will remove all their visit logs.`)) return;
        try {
            const res = await fetch(`/api/stakeholders/${uid}`, { method: 'DELETE' });
            if (res.ok) {
                showToast('Deleted', `Stakeholder ${name} removed from registry.`, 'info');
                loadStakeholdersTab();
                fetchStats();
            }
        } catch (err) {
            console.error(err);
        }
    };

    // -----------------------------------------------------------------------
    // Event Listeners & Tab Switching
    // -----------------------------------------------------------------------
    dom.navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            switchTab(targetTab);
        });
    });

    function switchTab(tabId) {
        state.activeTab = tabId;

        // Update nav buttons
        dom.navBtns.forEach(b => {
            b.classList.toggle('active', b.dataset.tab === tabId);
        });

        // Update panes
        dom.tabPanes.forEach(pane => {
            pane.classList.toggle('active', pane.id === `pane-${tabId}`);
        });

        // Update headers
        const titles = {
            live: ['Live Command Hub', 'Real-time deep learning surveillance & facial identification'],
            visits: ['Visit & Access History', 'Audited log of verified campus checkpoint detections'],
            unknowns: ['Unknown Persons Queue', 'Visual review of unauthorized or unregistered visitors'],
            stakeholders: ['Stakeholder Registry', 'Manage students, faculty, staff and authorized personnel biometric templates'],
            analytics: ['Surveillance Intelligence', 'Visual metrics, attendance flow and peak activity analytics'],
            settings: ['Surveillance Configuration', 'Input video sources, camera resolutions and threshold parameters']
        };

        if (titles[tabId]) {
            dom.pageTitle.textContent = titles[tabId][0];
            dom.pageSubtitle.textContent = titles[tabId][1];
        }

        // Trigger tab-specific loaders
        if (tabId === 'visits') loadVisitsTab();
        if (tabId === 'unknowns') loadUnknownsTab();
        if (tabId === 'stakeholders') loadStakeholdersTab();
        if (tabId === 'analytics') loadAnalyticsTab();
        if (tabId === 'settings') loadSettingsTab();

        if (window.lucide) lucide.createIcons();
    }

    // Live Snapshot
    dom.btnSnapshot.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = `surveillance_snapshot_${Date.now()}.jpg`;
        if (dom.liveVideoCanvas && dom.liveVideoCanvas.style.display !== 'none') {
            link.href = dom.liveVideoCanvas.toDataURL('image/jpeg', 0.95);
        } else {
            link.href = dom.liveVideoFeed.src;
        }
        link.click();
        showToast('Snapshot Saved', 'Current camera frame downloaded.', 'info');
    });

    // Fullscreen
    dom.btnFullscreen.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            dom.videoContainer.requestFullscreen().catch(err => alert(err.message));
        } else {
            document.exitFullscreen();
        }
    });

    // Pipeline Start/Stop Buttons
    dom.btnStartPipeline.addEventListener('click', () => startPipeline());
    dom.btnStopPipeline.addEventListener('click', () => stopPipeline());
    dom.btnOverlayStart.addEventListener('click', () => startPipeline());
    dom.btnQuickConfig.addEventListener('click', () => switchTab('settings'));

    // Visits search and filter
    dom.visitSearchInput.addEventListener('input', debounce(() => loadVisitsTab(), 300));
    dom.visitRoleFilter.addEventListener('change', () => loadVisitsTab());
    dom.btnExportVisits.addEventListener('click', () => {
        window.location.href = '/api/visits/export';
    });

    // Unknowns filters
    dom.chkOnlyUnverified.addEventListener('change', () => loadUnknownsTab());
    dom.btnRefreshUnknowns.addEventListener('click', () => loadUnknownsTab());

    // -----------------------------------------------------------------------
    // Stakeholder Registration Modal
    // -----------------------------------------------------------------------
    dom.btnOpenRegisterModal.addEventListener('click', () => {
        dom.modalRegister.classList.add('active');
    });

    function closeRegisterModal() {
        dom.modalRegister.classList.remove('active');
        dom.formRegisterStakeholder.reset();
        dom.photoPreviewContainer.style.display = 'none';
        dom.photoDropzoneText.textContent = 'Click or drag & drop a clear front-facing portrait photo';
    }

    dom.btnCloseRegisterModal.addEventListener('click', closeRegisterModal);
    dom.btnCancelRegister.addEventListener('click', closeRegisterModal);

    dom.photoDropzone.addEventListener('click', () => dom.regPhotoInput.click());

    dom.regPhotoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            dom.photoDropzoneText.textContent = `Selected: ${file.name}`;
            const reader = new FileReader();
            reader.onload = (re) => {
                dom.photoPreviewImg.src = re.target.result;
                dom.photoPreviewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    dom.formRegisterStakeholder.addEventListener('submit', async (e) => {
        e.preventDefault();
        const uid = document.getElementById('regUid').value.trim();
        const name = document.getElementById('regName').value.trim();
        const role = document.getElementById('regRole').value;
        const file = dom.regPhotoInput.files[0];

        if (!file) {
            alert('Please select a photo.');
            return;
        }

        const formData = new FormData();
        formData.append('uid', uid);
        formData.append('name', name);
        formData.append('role', role);
        formData.append('file', file);

        try {
            const res = await fetch('/api/stakeholders/register', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                showToast('Enrollment Complete', `Registered ${name} (${role})`, 'visit');
                closeRegisterModal();
                loadStakeholdersTab();
                fetchStats();
            } else {
                alert(data.detail || 'Registration failed.');
            }
        } catch (err) {
            alert('Registration error: ' + err.message);
        }
    });

    // -----------------------------------------------------------------------
    // Camera Settings Tab Handlers
    // -----------------------------------------------------------------------
    dom.inputSourceModes.forEach(radio => {
        radio.addEventListener('change', (e) => {
            const mode = e.target.value;
            dom.webcamConfigBox.style.display = mode === 'webcam' ? 'block' : 'none';
            dom.rtspConfigBox.style.display = mode === 'rtsp' ? 'block' : 'none';
            dom.videoConfigBox.style.display = mode === 'video' ? 'block' : 'none';
        });
    });

    // Video Upload Dropzone
    dom.videoDropzone.addEventListener('click', () => dom.videoFileInput.click());

    dom.videoFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        showToast('Uploading Video', `Uploading ${file.name}...`, 'info');
        try {
            const res = await fetch('/api/videos/upload', {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                showToast('Upload Complete', `${file.name} ready for playback.`, 'visit');
                await loadSettingsTab();
                dom.cfgVideoSelect.value = file.name;
            }
        } catch (err) {
            alert('Video upload failed: ' + err.message);
        }
    });

    dom.btnSaveApplyConfig.addEventListener('click', async () => {
        const mode = document.querySelector('input[name="inputSourceMode"]:checked').value;
        const location = dom.cfgLocationLabel.value.trim() || 'Campus Cam';
        let source = 0;

        if (mode === 'webcam') {
            source = parseInt(dom.cfgWebcamIndex.value) || 0;
        } else if (mode === 'rtsp') {
            source = dom.cfgRtspUrl.value.trim();
            if (!source) {
                alert('Please enter an RTSP URL.');
                return;
            }
        } else if (mode === 'video') {
            const videoFile = dom.cfgVideoSelect.value;
            if (!videoFile) {
                alert('Please select or upload a video file.');
                return;
            }
            source = `data/videos/${videoFile}`;
        }

        // Restart pipeline with new settings
        await stopPipeline();
        setTimeout(() => {
            startPipeline(source, location);
            switchTab('live');
        }, 800);
    });

    // Helper debounce utility
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // -----------------------------------------------------------------------
    // Initial Boot
    // -----------------------------------------------------------------------
    connectWebSocket();
    fetchStats();
    setInterval(fetchStats, 5000);
});
