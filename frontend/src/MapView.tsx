import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet-draw";
import type { GeoJsonObject } from "geojson";

export type GeoJson = GeoJsonObject;

export type MapRasterLayer = {
  id: string;
  name: string;
  task_name: string;
};

type MapViewProps = {
  defaultAoi?: GeoJson;
  focusAoi?: GeoJson | null;
  clearRequest: number;
  rasterLayers: MapRasterLayer[];
  opacity: number;
  onRasterStatusChange: (status: { loading: boolean; error: string | null }) => void;
  searchTarget: { lat: number; lon: number } | null;
  onAoiChange: (aoi: GeoJson | null) => void;
  onMapClick: (point: { lat: number; lon: number }) => void;
};

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function MapView({
  defaultAoi,
  focusAoi,
  clearRequest,
  rasterLayers,
  opacity,
  onRasterStatusChange,
  searchTarget,
  onAoiChange,
  onMapClick,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const drawnRef = useRef<L.FeatureGroup>(L.featureGroup());
  const onAoiChangeRef = useRef(onAoiChange);
  const onMapClickRef = useRef(onMapClick);
  const tileLayersRef = useRef<L.LayerGroup>(L.layerGroup());
  const rectangleStartRef = useRef<() => void>(() => {});
  const rectangleCancelRef = useRef<() => void>(() => {});
  const rectangleModeRef = useRef(false);
  const firstCornerRef = useRef<L.LatLng | null>(null);
  const dragStartRef = useRef<{ latlng: L.LatLng; point: L.Point } | null>(null);
  const ignoreClickUntilRef = useRef(0);
  const leafletDrawingRef = useRef(false);
  const [rectangleMode, setRectangleMode] = useState(false);
  const [rectangleHint, setRectangleHint] = useState("");

  useEffect(() => {
    onAoiChangeRef.current = onAoiChange;
    onMapClickRef.current = onMapClick;
  }, [onAoiChange, onMapClick]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      zoomControl: false,
      preferCanvas: true,
    }).setView([20, 0], 2);
    mapRef.current = map;

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(map);
    L.control.zoom({ position: "topright" }).addTo(map);
    drawnRef.current.addTo(map);
    tileLayersRef.current.addTo(map);
    const drawingPreview = L.featureGroup().addTo(map);

    const drawControl = new L.Control.Draw({
      position: "topleft",
      draw: {
        polygon: {
          allowIntersection: false,
          showArea: true,
          shapeOptions: { color: "#23846b", weight: 2 },
        },
        rectangle: false,
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: { featureGroup: drawnRef.current, remove: true },
    });
    map.addControl(drawControl);

    const finishRectangle = (bounds: L.LatLngBounds) => {
      drawingPreview.clearLayers();
      const rectangle = L.rectangle(bounds, {
        color: "#23846b",
        weight: 2,
        fillOpacity: 0.1,
      });
      drawnRef.current.clearLayers();
      drawnRef.current.addLayer(rectangle);
      onAoiChangeRef.current(rectangle.toGeoJSON());
      rectangleModeRef.current = false;
      firstCornerRef.current = null;
      dragStartRef.current = null;
      map.dragging.enable();
      map.getContainer().classList.remove("leaflet-rectangle-mode");
      setRectangleMode(false);
      setRectangleHint("");
    };

    const cancelRectangle = () => {
      rectangleModeRef.current = false;
      firstCornerRef.current = null;
      dragStartRef.current = null;
      drawingPreview.clearLayers();
      map.dragging.enable();
      map.getContainer().classList.remove("leaflet-rectangle-mode");
      setRectangleMode(false);
      setRectangleHint("");
    };

    rectangleStartRef.current = () => {
      if (rectangleModeRef.current) {
        cancelRectangle();
        return;
      }
      rectangleModeRef.current = true;
      firstCornerRef.current = null;
      map.dragging.disable();
      map.getContainer().classList.add("leaflet-rectangle-mode");
      setRectangleMode(true);
      setRectangleHint("Click the first corner, then the opposite corner; or drag.");
    };
    rectangleCancelRef.current = cancelRectangle;

    map.on(L.Draw.Event.DRAWSTART, () => {
      leafletDrawingRef.current = true;
    });
    map.on(L.Draw.Event.DRAWSTOP, () => {
      leafletDrawingRef.current = false;
    });
    map.on("mousedown", (event: L.LeafletMouseEvent) => {
      if (!rectangleModeRef.current) return;
      const start = firstCornerRef.current ?? event.latlng;
      dragStartRef.current = {
        latlng: start,
        point: map.latLngToContainerPoint(start),
      };
    });
    map.on("mousemove", (event: L.LeafletMouseEvent) => {
      const start = dragStartRef.current;
      if (!rectangleModeRef.current || !start) return;
      const point = map.latLngToContainerPoint(event.latlng);
      if (point.distanceTo(start.point) < 5) return;
      drawingPreview.clearLayers();
      L.rectangle(L.latLngBounds(start.latlng, event.latlng), {
        color: "#23846b",
        weight: 2,
        dashArray: "5 4",
        fillOpacity: 0.08,
      }).addTo(drawingPreview);
      setRectangleHint("Release to finish the rectangle.");
    });
    map.on("mouseup", (event: L.LeafletMouseEvent) => {
      const start = dragStartRef.current;
      if (!rectangleModeRef.current || !start) return;
      const endPoint = map.latLngToContainerPoint(event.latlng);
      if (endPoint.distanceTo(start.point) < 5) return;
      ignoreClickUntilRef.current = Date.now() + 250;
      finishRectangle(L.latLngBounds(start.latlng, event.latlng));
    });

    map.on(L.Draw.Event.CREATED, (event: L.LeafletEvent) => {
      const created = event as L.LeafletEvent & { layer: L.Layer & { toGeoJSON: () => GeoJson } };
      drawnRef.current.clearLayers();
      drawnRef.current.addLayer(created.layer);
      onAoiChangeRef.current(created.layer.toGeoJSON());
    });
    map.on(L.Draw.Event.EDITED, () => {
      const layer = drawnRef.current.getLayers()[0] as L.Layer & {
        toGeoJSON: () => GeoJson;
      };
      if (layer) onAoiChangeRef.current(layer.toGeoJSON());
    });
    map.on(L.Draw.Event.DELETED, () => onAoiChangeRef.current(null));
    map.on("click", (event: L.LeafletMouseEvent) => {
      if (Date.now() < ignoreClickUntilRef.current) {
        return;
      }
      if (rectangleModeRef.current) {
        dragStartRef.current = null;
        const firstCorner = firstCornerRef.current;
        if (!firstCorner) {
          firstCornerRef.current = event.latlng;
          drawingPreview.clearLayers();
          L.circleMarker(event.latlng, {
            radius: 5,
            color: "#176650",
            weight: 2,
            fillColor: "#fff",
            fillOpacity: 1,
          }).addTo(drawingPreview);
          setRectangleHint("First corner set. Click the opposite corner or drag.");
          return;
        }
        const bounds = L.latLngBounds(firstCorner, event.latlng);
        const firstPoint = map.latLngToContainerPoint(firstCorner);
        const secondPoint = map.latLngToContainerPoint(event.latlng);
        if (firstPoint.distanceTo(secondPoint) < 5) {
          setRectangleHint("Choose a different point for the opposite corner.");
          return;
        }
        finishRectangle(bounds);
        return;
      }
      if (leafletDrawingRef.current) return;
      onMapClickRef.current({ lat: event.latlng.lat, lon: event.latlng.lng });
    });
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && rectangleModeRef.current) cancelRectangle();
    };
    document.addEventListener("keydown", onKeyDown);

    if (defaultAoi) {
      const area = L.geoJSON(defaultAoi, {
        style: { color: "#23846b", weight: 2, fillOpacity: 0.1 },
      });
      area.eachLayer((layer) => drawnRef.current.addLayer(layer));
      const bounds = area.getBounds();
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40] });
    }

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      rectangleCancelRef.current = () => {};
      rectangleStartRef.current = () => {};
      map.remove();
      mapRef.current = null;
    };
  }, [defaultAoi]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !searchTarget) return;
    map.flyTo([searchTarget.lat, searchTarget.lon], Math.max(map.getZoom(), 11));
  }, [searchTarget]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !focusAoi) return;
    const area = L.geoJSON(focusAoi, {
      style: { color: "#23846b", weight: 2, fillOpacity: 0.1 },
    });
    drawnRef.current.clearLayers();
    area.eachLayer((layer) => drawnRef.current.addLayer(layer));
    const bounds = area.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40] });
  }, [focusAoi]);

  useEffect(() => {
    if (clearRequest > 0) {
      drawnRef.current.clearLayers();
      rectangleCancelRef.current();
    }
  }, [clearRequest]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    tileLayersRef.current.clearLayers();
    const loadingLayers = new Set<string>();
    let tileError: string | null = null;
    onRasterStatusChange({ loading: false, error: null });
    rasterLayers.forEach((layer, index) => {
      const tileLayer = L.tileLayer(
        `${API_BASE}/api/layers/${layer.id}/tiles/{z}/{x}/{y}.png?task_name=${encodeURIComponent(layer.task_name)}`,
        { opacity, tileSize: 256, maxZoom: 19, zIndex: 300 + index },
      );
      tileLayer.on("loading", () => {
        loadingLayers.add(layer.id);
        onRasterStatusChange({ loading: true, error: tileError });
      });
      tileLayer.on("load", () => {
        loadingLayers.delete(layer.id);
        onRasterStatusChange({ loading: loadingLayers.size > 0, error: tileError });
      });
      tileLayer.on("tileerror", () => {
        tileError = `Could not load map tiles for ${layer.name}. Check the task data and API status.`;
        loadingLayers.delete(layer.id);
        onRasterStatusChange({ loading: loadingLayers.size > 0, error: tileError });
      });
      tileLayer.addTo(tileLayersRef.current);
    });
    if (!map.hasLayer(tileLayersRef.current)) tileLayersRef.current.addTo(map);
    return () => {
      tileLayersRef.current.clearLayers();
    };
  }, [rasterLayers, opacity, onRasterStatusChange]);

  return (
    <>
      <div className="map-canvas" ref={containerRef} aria-label="Interactive map" />
      <div className="map-draw-rectangle">
        <button
          type="button"
          className={rectangleMode ? "active" : ""}
          aria-pressed={rectangleMode}
          onClick={() => rectangleStartRef.current()}
        >
          <span aria-hidden="true">▱</span>
          {rectangleMode ? "Cancel rectangle" : "Draw rectangle"}
        </button>
        {rectangleHint && <span role="status">{rectangleHint}</span>}
      </div>
    </>
  );
}
