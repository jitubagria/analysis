// Ensure Streamlit is available
const Streamlit = window.parent.Streamlit;

let currentColumns = [];
const dropZoneIds = ['x-axis-dropzone', 'y-axis-dropzone', 'color-dropzone']; // Add other IDs

function initializeDragAndDrop() {
    const columnsSourceEl = document.getElementById('columns-source');
    columnsSourceEl.innerHTML = ''; // Clear previous items

    currentColumns.forEach(col => {
        const item = document.createElement('div');
        item.className = 'draggable-item';
        item.textContent = col;
        item.dataset.columnName = col; // Store column name
        columnsSourceEl.appendChild(item);
    });

    // Initialize SortableJS for the source list (cloning items)
    new Sortable(columnsSourceEl, {
        group: {
            name: 'shared',
            pull: 'clone', // Clone items so they remain in the source list
            put: false // Do not allow items to be put back into the source list directly
        },
        sort: false, // Don't sort the source list
        animation: 150
    });

    // Initialize SortableJS for each drop zone
    dropZoneIds.forEach(zoneId => {
        const zoneEl = document.getElementById(zoneId);
        if (zoneEl) {
            new Sortable(zoneEl, {
                group: 'shared', // Allow items to be dropped from 'shared' group
                animation: 150,
                onAdd: function (evt) {
                    // Limit single-item zones
                    if (zoneEl.classList.contains('single-item-zone') && zoneEl.children.length > 1) {
                        // If more than one item, remove the previously added one(s) except the new one
                        while (zoneEl.children.length > 1 && zoneEl.firstChild !== evt.item) {
                            columnsSourceEl.appendChild(zoneEl.firstChild); // Move extra back to source or just remove
                        }
                    }
                    sendConfigToPython();
                },
                onRemove: function (evt) {
                    sendConfigToPython();
                },
                // onUpdate for reordering if needed within multi-item zones
            });
        }
    });
}

function getItemsInZone(zoneId) {
    const zoneEl = document.getElementById(zoneId);
    if (!zoneEl) return [];
    return Array.from(zoneEl.children).map(item => item.dataset.columnName);
}

function sendConfigToPython() {
    const config = {
        x_axis: getItemsInZone('x-axis-dropzone')[0] || null, // Takes the first item or null
        y_axes: getItemsInZone('y-axis-dropzone'),          // Takes all items as a list
        color: getItemsInZone('color-dropzone')[0] || null,
        // size: getItemsInZone('size-dropzone')[0] || null,
        // facet_row: getItemsInZone('facet-row-dropzone')[0] || null,
        // facet_col: getItemsInZone('facet-col-dropzone')[0] || null,
    };
    Streamlit.setComponentValue(config);
}

function onRender(event) {
    const data = event.detail;
    if (data.args.columns) {
        currentColumns = data.args.columns;
        initializeDragAndDrop();
    }
    Streamlit.setFrameHeight(); // Auto-adjust height of the iframe
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(); // Initial height adjustment