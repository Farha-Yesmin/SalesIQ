/* Resolves API and page URLs when the HTML is not served from Flask (e.g. file://
   or Live Server on another port). When served from http://127.0.0.1:5000, paths stay relative. */
(function () {
    var FLASK_PORT = '5000';
    var loc = window.location;
    var base = '';
    if (loc.protocol === 'file:') {
        base = 'http://127.0.0.1:' + FLASK_PORT;
    } else if (loc.protocol === 'http:' || loc.protocol === 'https:') {
        var local = loc.hostname === '127.0.0.1' || loc.hostname === 'localhost';
        if (local && loc.port && loc.port !== FLASK_PORT) {
            base = loc.protocol + '//' + loc.hostname + ':' + FLASK_PORT;
        }
    }
    window.salesiqApiBase = base;
    window.salesiqUrl = function (path) {
        return base + path;
    };
    function patchLocalLinks() {
        if (!base) return;
        document.querySelectorAll('a[href^="/"]').forEach(function (a) {
            var h = a.getAttribute('href');
            if (h && h.charAt(0) === '/' && h.indexOf('//') !== 0) {
                a.setAttribute('href', base + h);
            }
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', patchLocalLinks);
    } else {
        patchLocalLinks();
    }

    var CACHE_KEY = 'salesiq_result';

    window.salesiqClearResultCache = function () {
        try { sessionStorage.removeItem(CACHE_KEY); } catch (e) { /* ignore */ }
    };

    /** Drop cache if it was saved by a different logged-in user. */
    window.salesiqReadResultCache = function (currentEmail) {
        try {
            var raw = sessionStorage.getItem(CACHE_KEY);
            if (!raw) return null;
            var p = JSON.parse(raw);
            if (currentEmail) {
                if (!p._salesiqUser || p._salesiqUser !== currentEmail) {
                    sessionStorage.removeItem(CACHE_KEY);
                    return null;
                }
            }
            return p;
        } catch (e) {
            return null;
        }
    };

    /**
     * Clone the object before tagging it so we never mutate the caller's data.
     */
    window.salesiqWriteResultCache = function (obj, email) {
        try {
            var toStore = JSON.parse(JSON.stringify(obj)); // deep clone — no mutation
            if (email) toStore._salesiqUser = email;
            sessionStorage.setItem(CACHE_KEY, JSON.stringify(toStore));
        } catch (e) { /* sessionStorage full or unavailable — ignore */ }
    };

    /**
     * Always fetch /api/results/latest first (source of truth).
     *
     * KEY FIX: After a new upload, the DB run_date will differ from any cached
     * run_date. We now explicitly discard the stale cache when run_date doesn't
     * match, so re-uploads always display fresh results on every page.
     *
     * historicalQty is only merged from cache when the run_date matches exactly
     * (same prediction run, DB rows simply lack the field).
     */
    window.salesiqHydrateLatest = async function (userEmail) {
        try {
            var res = await fetch(window.salesiqUrl('/api/results/latest'), { credentials: 'include' });
            var text = await res.text();
            var json = {};
            try {
                json = text ? JSON.parse(text) : {};
            } catch (e1) {
                return {
                    ok: false,
                    data: null,
                    message: 'Server returned an invalid response. Check that Flask is running at http://127.0.0.1:5000',
                };
            }
            if (!res.ok) {
                return {
                    ok: false,
                    data: null,
                    message: (json && json.error) ? json.error : ('Request failed (' + res.status + '). Is Flask running?'),
                };
            }
            if (!json.has_data) {
                window.salesiqClearResultCache();
                return { ok: true, data: null };
            }

            /* Read whatever is currently in cache */
            var cached = null;
            try {
                var raw = sessionStorage.getItem(CACHE_KEY);
                if (raw) cached = JSON.parse(raw);
            } catch (e2) { /* ignore */ }

            /*
             * Guard 1 - wrong user: drop cache entirely.
             * Guard 2 - stale run: the DB has a newer prediction (different
             *   run_date). Drop the old cache so the UI always reflects the
             *   latest upload, not a previous one.
             * Only after both guards pass do we attempt the historicalQty merge.
             */
            if (cached) {
                var sameUser = !userEmail ||
                    !cached._salesiqUser ||
                    cached._salesiqUser === userEmail;

                if (!sameUser || cached.run_date !== json.run_date) {
                    /* Stale or foreign cache - discard it */
                    window.salesiqClearResultCache();
                    cached = null;
                }
            }

            /* Merge historicalQty from same-run cache when DB rows lack it */
            if (cached && json.products && cached.products) {
                json.products = json.products.map(function (p) {
                    var h = p.historicalQty || [];
                    var has = h.some(function (v) { return Number(v) > 0; });
                    if (has) return p;
                    var c = cached.products.find(function (x) {
                        return (x.product || x.product_name) === p.product;
                    });
                    if (c && c.historicalQty && c.historicalQty.some(function (v) { return Number(v) > 0; })) {
                        var o = JSON.parse(JSON.stringify(p)); // clone before mutating
                        o.historicalQty = c.historicalQty.slice();
                        return o;
                    }
                    return p;
                });
            }

            /* Always write the latest result back to cache */
            window.salesiqWriteResultCache(json, userEmail);
            return { ok: true, data: json };
        } catch (e) {
            return {
                ok: false,
                data: null,
                message: 'Cannot reach the server. Start the app with: python app.py then open http://127.0.0.1:5000',
            };
        }
    };
})();