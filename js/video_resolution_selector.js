import { app } from "../../scripts/app.js";

const NODE_CLASS = "VideoResolutionSelector";
const HIDDEN_WIDGET_TYPE = "converted-widget";

function setWidgetVisible(widget, visible) {
    if (!widget) {
        return;
    }

    if (!widget.__omarioOriginalState) {
        widget.__omarioOriginalState = {
            type: widget.type,
            computeSize: widget.computeSize,
        };
    }

    if (visible) {
        widget.type = widget.__omarioOriginalState.type;
        widget.computeSize = widget.__omarioOriginalState.computeSize;
    } else {
        widget.type = HIDDEN_WIDGET_TYPE;
        widget.computeSize = () => [0, -4];
    }
}

function updateModeWidgets(node) {
    const modeWidget = node.widgets?.find((widget) => widget.name === "size_mode");
    const megapixelsWidget = node.widgets?.find(
        (widget) => widget.name === "megapixels",
    );
    const maxPixelsWidget = node.widgets?.find(
        (widget) => widget.name === "max_pixels",
    );
    const shorterPixelsWidget = node.widgets?.find(
        (widget) => widget.name === "shorter_pixels",
    );

    if (
        !modeWidget ||
        !megapixelsWidget ||
        !maxPixelsWidget ||
        !shorterPixelsWidget
    ) {
        return;
    }

    setWidgetVisible(megapixelsWidget, modeWidget.value === "Megapixels");
    setWidgetVisible(maxPixelsWidget, modeWidget.value === "Max Pixels");
    setWidgetVisible(shorterPixelsWidget, modeWidget.value === "Shorter Size");

    const computedSize = node.computeSize();
    node.setSize([Math.max(node.size[0], computedSize[0]), computedSize[1]]);
    node.graph?.setDirtyCanvas(true, true);
}

function configureNode(node) {
    const comfyClass = node.comfyClass ?? node.constructor?.ComfyClass;
    if (comfyClass !== NODE_CLASS) {
        return;
    }

    const modeWidget = node.widgets?.find((widget) => widget.name === "size_mode");
    if (!modeWidget) {
        return;
    }

    if (!modeWidget.__omarioVisibilityCallback) {
        const originalCallback = modeWidget.callback;
        modeWidget.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            updateModeWidgets(node);
            return result;
        };
        modeWidget.__omarioVisibilityCallback = true;
    }

    updateModeWidgets(node);
}

app.registerExtension({
    name: "Omario.VideoResolutionSelector.WidgetVisibility",

    nodeCreated(node) {
        configureNode(node);
    },

    loadedGraphNode(node) {
        configureNode(node);
    },
});
