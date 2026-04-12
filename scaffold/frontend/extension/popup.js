const hSlider = document.getElementById("h-slider");
const eSlider = document.getElementById("e-slider");
const hVal = document.getElementById("h-val");
const eVal = document.getElementById("e-val");

// Load saved thresholds
chrome.runtime.sendMessage(
	{ target: "background", type: "get-thresholds" },
	(t) => {
		if (t) {
			hSlider.value = t.habermas;
			hVal.textContent = t.habermas;
			eSlider.value = t.erscheinung;
			eVal.textContent = t.erscheinung;
		}
	},
);

hSlider.addEventListener("input", () => {
	hVal.textContent = hSlider.value;
	chrome.runtime.sendMessage({
		target: "background",
		type: "set-thresholds",
		thresholds: {
			habermas: parseInt(hSlider.value),
			erscheinung: parseInt(eSlider.value),
		},
	});
});

eSlider.addEventListener("input", () => {
	eVal.textContent = eSlider.value;
	chrome.runtime.sendMessage({
		target: "background",
		type: "set-thresholds",
		thresholds: {
			habermas: parseInt(hSlider.value),
			erscheinung: parseInt(eSlider.value),
		},
	});
});

// V2: Layer toggles
const pauseToggle = document.getElementById("pause-toggle");
const tagsToggle = document.getElementById("tags-toggle");

chrome.storage.local.get({ pauseEnabled: true, tagsEnabled: true }, (t) => {
	pauseToggle.checked = t.pauseEnabled;
	tagsToggle.checked = t.tagsEnabled;
});

pauseToggle.addEventListener("change", () => {
	chrome.storage.local.set({ pauseEnabled: pauseToggle.checked });
});

tagsToggle.addEventListener("change", () => {
	chrome.storage.local.set({ tagsEnabled: tagsToggle.checked });
});
