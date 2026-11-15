import {
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import type {
  Feature,
  FeatureCollection,
  Geometry,
  GeometryCollection,
} from "geojson";
import MapView, { type GeoJson } from "./MapView";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type Layer = {
  layer_id: string;
  name: string;
  category: string;
  description?: string | null;
  units?: string | null;
  spatial_resolution?: string | null;
  temporal_resolution?: string | null;
  available: boolean;
  is_quality: boolean;
  quality_layers: string[];
};

type EssentialVariable = {
  id: string;
  name: string;
  product_id?: string | null;
  layer_id?: string | null;
  units?: string | null;
  spatial_resolution?: string | null;
  temporal_resolution?: string | null;
  temporal_start?: string | null;
  temporal_end?: string | null;
  scale_factor?: number | null;
  add_offset?: number | null;
  fill_values: string[];
  raw_valid_range?: [string | number | null, string | number | null] | null;
  valid_range?: [string | number | null, string | number | null] | null;
  quality_layers: string[];
  qa_required: boolean;
  requestable_from_saved_metadata: boolean;
  processed_on_selected_task: boolean;
  availability_status: string;
  availability_detail?: string | null;
  data_scope: string;
};

type EssentialLayer = {
  id: string;
  name: string;
  category: string;
  description: string;
  source: string;
  required_for_suitability: boolean;
  derived: boolean;
  variables: EssentialVariable[];
  requestable_from_saved_metadata: boolean;
  processed_on_selected_task: boolean;
  availability_status: string;
};

type SuitabilityInputs = {
  map_visibility_changes_analysis_inputs: boolean;
  crop_reference_scoring_criteria: string[];
  matching_raster_scoring_criteria: string[];
  status: string;
};

type EssentialCatalogResponse = {
  layers: EssentialLayer[];
  suitability_inputs: SuitabilityInputs;
};

type Product = {
  product_id: string;
  name: string;
  description?: string | null;
  spatial_resolution?: string | null;
  temporal_resolution?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  layers: Layer[];
};

type Settings = {
  analysis_crs: string;
  resolution_m: number;
  max_grid_cells: number;
  start_date: string;
  end_date: string;
};

type StoredAoi = {
  name: string;
  crs: string;
  geojson: GeoJson;
};

type AoiStats = {
  name: string;
  geometry_type: string;
  bounds: { west: number; south: number; east: number; north: number };
  area_km2: number;
  estimated_grid_cells: number;
  analysis_crs: string;
  grid_limit_exceeded: boolean;
};

type Task = {
  task_name?: string | null;
  task_id: string | null;
  task_type?: string | null;
  product_id?: string | null;
  aoi_name?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  submitted_at?: string | null;
  task_status: string;
  download_status: string;
  preprocessing_status?: string;
  analysis_run_count?: number;
  downloaded_file_count: number;
  error?: string | null;
  remote_only?: boolean;
  acquisition?: {
    status: string;
    stage?: string;
    providers: Record<string, {
      status: string;
      error?: string;
      variables?: string[];
      failures?: Array<{ product_id: string; error: string }>;
      operations?: Array<{
        task_id: string;
        task_name?: string;
        product_id?: string;
        status: string;
        download_status?: string;
        error?: string | null;
      }>;
    }>;
  } | null;
};

type RasterLayer = {
  id: string;
  task_name: string;
  analysis_name?: string | null;
  product_id: string;
  variable: string;
  name: string;
  observation_date: string | null;
  temporal_start?: string | null;
  temporal_end?: string | null;
  units: string | null;
  variable_kind: string;
  quality_filter_applied: boolean;
  quality_layers: string[];
  render_range: [number, number] | null;
  category_colors?: Record<string, string> | null;
  category_labels?: Record<string, string> | null;
};

type Crop = { id: string; name: string };
type GaezCrop = {
  code: string;
  name: string;
  category: string;
  project_crop_id: string;
  availability_status: "available" | "partially_available";
  available_scenario_codes: string[];
};
type GaezUnavailableCrop = {
  id: string;
  name: string;
  category: string;
  availability_status: "unavailable" | "not_yet_integrated";
  availability_reason: string;
};
type GaezCropResult = {
  crop_code: string;
  crop_name: string;
  rank?: number;
  mean_suitability_index: number | null;
  suitability_index_units: string;
  valid_data_coverage: number;
  valid_data_coverage_of_aoi?: number;
  valid_data_coverage_basis?: string;
  aoi_area_km2?: number;
  water_area_km2?: number;
  water_area_share_of_aoi?: number;
  intersecting_cell_count?: number;
  valid_index_cell_count?: number;
  water_cell_count?: number;
  ranking_eligible: boolean;
  ranking_exclusion_reason?: string | null;
  class_data_available?: boolean;
  class_data_status?: string;
  class_data_error?: string | null;
  valid_class_cell_count?: number;
  class_data_coverage?: number;
  classified_area_km2?: number;
  class_distribution: Array<{
    class_code: number;
    label: string;
    area_km2: number;
    share_of_aoi?: number;
    share_of_classified_area: number | null;
  }>;
  continuous_map_layer: string;
  class_map_layer: string | null;
};
type GaezScenarioResult = {
  water_regime: "rainfed" | "irrigated";
  management_level: "low" | "high";
  management_code?: string;
  product_identifiers?: {
    continuous_index: string;
    official_classes: string;
  };
  ranked_crops: GaezCropResult[];
  unranked_crops: GaezCropResult[];
};
type GaezFailure = {
  crop_code: string;
  crop_name?: string;
  water_regime: string;
  management_level: string;
  product?: string;
  failure_stage?: string;
  failure_type?: string;
  error: string;
};
type GaezResult = {
  task_name: string;
  task_id: string;
  run_id?: string;
  analysis_version?: number;
  analysis_outdated?: boolean;
  status: string;
  operations?: Array<{
    crop_code: string;
    crop_name: string;
    water_regime: string;
    management_level?: string;
    status: string;
    failure_stage?: string;
    failure_type?: string;
    error?: string;
  }>;
  error?: string | null;
  period?: string;
  period_code?: string;
  climate_code?: string;
  scenario_code?: string;
  crs?: string;
  spatial_resolution_degrees?: number;
  minimum_coverage_for_ranking?: number;
  attribution?: string;
  legend_url?: string;
  continuous_product?: { mapset: string; dataset_id: string; url: string };
  class_product?: { mapset: string; dataset_id: string; url: string };
  index_metadata?: {
    units: string;
    valid_range: [number, number];
    scale_factor: number;
    no_data: number;
  };
  class_metadata?: {
    units: string;
    valid_range: [number, number];
    scale_factor: number;
    no_data: number;
  };
  failures?: GaezFailure[];
  limitations?: string[];
  scenarios?: GaezScenarioResult[];
};
type GaezCatalog = {
  provider: string;
  period: string;
  spatial_resolution_degrees: number;
  crs: string;
  minimum_coverage_for_ranking: number;
  crops: GaezCrop[];
  unavailable_crops: GaezUnavailableCrop[];
  inventory_snapshot: {
    verified_on: string;
    period_code: string;
    climate_code: string;
    scenario_code: string;
  };
  source: string;
  attribution: string;
};
type AppNotice = { kind: "error" | "success" | "info"; message: string };

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function objectRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function formatCropScores(value: unknown, crops: Crop[]): string {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "";
  return Object.entries(value as Record<string, unknown>)
    .map(([cropId, score]) => {
      const name = crops.find((crop) => crop.id === cropId)?.name ?? cropId;
      return `${name}: ${score == null ? "missing data" : score}`;
    })
    .join(" · ");
}

function formatLimitingFactors(value: unknown, crops: Crop[]): string {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "";
  return Object.entries(value as Record<string, unknown>)
    .flatMap(([cropId, entries]) => {
      if (!Array.isArray(entries)) return [];
      const cropName = crops.find((crop) => crop.id === cropId)?.name ?? cropId;
      return entries.flatMap((entry) => {
        if (!entry || typeof entry !== "object") return [];
        const factor = entry as Record<string, unknown>;
        if (typeof factor.parameter !== "string") return [];
        const score = typeof factor.score === "number" ? ` (${factor.score})` : "";
        return [`${cropName}: ${factor.parameter}${score}`];
      });
    })
    .join(" · ");
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      typeof payload.detail === "string"
        ? payload.detail
        : `Request failed (${response.status}).`,
    );
  }
  return payload as T;
}

function layerKey(productId: string, layerId: string) {
  return `${productId}::${layerId}`;
}

const ACQUISITION_PROVIDER_IDS = [
  "soil_properties",
  "nasa_power_temperature",
  "nasa_power_precipitation",
  "gaez_v5",
];

function aoiGeometries(geojson: GeoJson): unknown[] {
  if (geojson.type === "Feature") {
    return [(geojson as Feature<Geometry>).geometry];
  }
  if (geojson.type === "FeatureCollection") {
    return (geojson as FeatureCollection<Geometry>).features.map(
      (feature) => feature.geometry,
    );
  }
  if (geojson.type === "GeometryCollection") {
    return (geojson as GeometryCollection).geometries;
  }
  return [geojson];
}

function sameAoiGeometry(selected: GeoJson, stored: GeoJson): boolean {
  return JSON.stringify(aoiGeometries(selected)) === JSON.stringify(aoiGeometries(stored));
}

function App() {
  const [essentialLayers, setEssentialLayers] = useState<EssentialLayer[]>([]);
  const [essentialLayersError, setEssentialLayersError] = useState<string | null>(null);
  const [suitabilityInputs, setSuitabilityInputs] = useState<SuitabilityInputs | null>(null);
  const [aoi, setAoi] = useState<GeoJson | null>(null);
  const [mapFocusAoi, setMapFocusAoi] = useState<GeoJson | null>(null);
  const [aoiStats, setAoiStats] = useState<AoiStats | null>(null);
  const [aoiName, setAoiName] = useState("Selected area");
  const [clearAoiVersion, setClearAoiVersion] = useState(0);
  const [aoiError, setAoiError] = useState<string | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskName, setSelectedTaskName] = useState("");
  const [rasters, setRasters] = useState<RasterLayer[]>([]);
  const [crops, setCrops] = useState<Crop[]>([]);
  const [gaezCatalog, setGaezCatalog] = useState<GaezCatalog | null>(null);
  const [gaezResult, setGaezResult] = useState<GaezResult | null>(null);
  const [gaezCropCodes, setGaezCropCodes] = useState<Set<string>>(new Set());
  const [gaezCropSearch, setGaezCropSearch] = useState("");
  const [gaezWaterRegime, setGaezWaterRegime] = useState<"rainfed" | "irrigated" | "both">("both");
  const [gaezManagementLevel, setGaezManagementLevel] = useState<"low" | "high">("high");
  const [gaezResultRegime, setGaezResultRegime] = useState<"rainfed" | "irrigated">("rainfed");
  const [suitabilityAvailable, setSuitabilityAvailable] = useState(false);
  const [analysisName, setAnalysisName] = useState("");
  const [activeAnalysisName, setActiveAnalysisName] = useState<string | null>(null);
  const [selectedCropIds, setSelectedCropIds] = useState<Set<string>>(new Set());
  const [selectedScoreCropId, setSelectedScoreCropId] = useState("");
  const [suitabilityMode, setSuitabilityMode] = useState<"best-crop" | "individual-score">("best-crop");
  const [selectedLayers, setSelectedLayers] = useState<Set<string>>(new Set());
  const [selectedProviders, setSelectedProviders] = useState<Set<string>>(new Set());
  const [visibleRasters, setVisibleRasters] = useState<Set<string>>(new Set());
  const [layerOrder, setLayerOrder] = useState<string[]>([]);
  const [tileStatus, setTileStatus] = useState<{ loading: boolean; error: string | null }>({
    loading: false,
    error: null,
  });
  const [rastersLoading, setRastersLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [searchTarget, setSearchTarget] = useState<{
    lat: number;
    lon: number;
  } | null>(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [taskName, setTaskName] = useState("");
  const [analysisCrs, setAnalysisCrs] = useState("EPSG:6933");
  const [resolution, setResolution] = useState(250);
  const [opacity, setOpacity] = useState(0.78);
  const [inspection, setInspection] = useState<Record<string, unknown> | null>(
    null,
  );
  const [taskDetails, setTaskDetails] = useState<Record<string, unknown> | null>(
    null,
  );
  const [notice, setNotice] = useState<AppNotice | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [soilDepth, setSoilDepth] = useState("0-5cm");
  const [activeTab, setActiveTab] = useState("Map");

  const gaezCropGroups = useMemo(() => {
    const searchTerm = gaezCropSearch.trim().toLocaleLowerCase();
    const groups = new Map<string, Array<GaezCrop | GaezUnavailableCrop>>();
    for (const crop of [
      ...(gaezCatalog?.crops ?? []),
      ...(gaezCatalog?.unavailable_crops ?? []),
    ]) {
      if (
        searchTerm &&
        !`${crop.name} ${crop.category}`.toLocaleLowerCase().includes(searchTerm)
      ) {
        continue;
      }
      const group = groups.get(crop.category) ?? [];
      group.push(crop);
      groups.set(crop.category, group);
    }
    return [...groups.entries()].sort(([first], [second]) =>
      first.localeCompare(second),
    );
  }, [gaezCatalog, gaezCropSearch]);

  const refreshTasks = useCallback(async (includeRemote = false) => {
    const result = await requestJson<{ tasks: Task[]; warnings?: string[] }>(
      `/api/tasks${includeRemote ? "?include_remote=true" : ""}`,
    );
    setTasks(result.tasks);
    if (result.warnings?.length) {
      setNotice({
        kind: "info",
        message: `Some unavailable task records were omitted: ${result.warnings.join(" ")}`,
      });
    }
  }, []);

  const refreshRasters = useCallback(async (taskName: string, runName?: string | null) => {
    const query = new URLSearchParams({ task_name: taskName });
    if (runName) query.set("analysis_name", runName);
    const result = await requestJson<{ layers: RasterLayer[] }>(`/api/layers?${query}`);
    return result.layers;
  }, []);

  useEffect(() => {
    let active = true;
    requestJson<GaezCatalog>("/api/gaez-v5/catalog")
      .then((catalog) => {
        if (active) {
          setGaezCatalog(catalog);
          setGaezCropCodes((current) =>
            current.size ? current : new Set(catalog.crops.map((crop) => crop.code)),
          );
        }
      })
      .catch((error: Error) => {
        if (active) {
          setNotice({ kind: "error", message: `GAEZ v5 catalog is unavailable: ${error.message}` });
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const taskId = tasks.find((task) => task.task_name === selectedTaskName)?.task_id;
    if (!taskId) {
      setGaezResult(null);
      return;
    }
    let active = true;
    requestJson<GaezResult>(
      `/api/aoi-tasks/${encodeURIComponent(taskId)}/gaez-suitability`,
    )
      .then((result) => {
        if (active) setGaezResult(result);
      })
      .catch(() => {
        if (active) setGaezResult(null);
      });
    return () => {
      active = false;
    };
  }, [selectedTaskName, tasks]);

  useEffect(() => {
    let active = true;
    Promise.allSettled([
      requestJson<EssentialCatalogResponse>("/api/essential-layers"),
      requestJson<Settings>("/api/settings"),
      requestJson<{ tasks: Task[]; warnings?: string[] }>("/api/tasks"),
      requestJson<{ crops: Crop[]; suitability_available: boolean }>("/api/crops"),
    ])
      .then(([catalogResult, settingsResult, taskResult, cropResult]) => {
        if (!active) return;
        if (catalogResult.status === "fulfilled") {
          setEssentialLayers(catalogResult.value.layers);
          setSuitabilityInputs(catalogResult.value.suitability_inputs);
          setEssentialLayersError(null);
          setSelectedLayers(new Set(
            catalogResult.value.layers.flatMap((logicalLayer) =>
              logicalLayer.variables.flatMap((variable) =>
                variable.requestable_from_saved_metadata
                  && variable.product_id
                  && variable.layer_id
                  ? [layerKey(variable.product_id, variable.layer_id)]
                  : [],
              ),
            ),
          ));
          setSelectedProviders(new Set(ACQUISITION_PROVIDER_IDS));
        } else {
          const errorMessage = getErrorMessage(catalogResult.reason);
          setEssentialLayersError(errorMessage);
          setNotice({
            kind: "error",
            message: errorMessage === "Not Found"
              ? "The API server is outdated. Restart this project's backend to enable the curated layer catalog."
              : errorMessage,
          });
        }
        if (settingsResult.status === "fulfilled") {
          const settingsData = settingsResult.value;
          setStartDate(settingsData.start_date);
          setEndDate(settingsData.end_date);
          setAnalysisCrs(settingsData.analysis_crs);
          setResolution(settingsData.resolution_m);
        } else {
          setNotice({ kind: "error", message: getErrorMessage(settingsResult.reason) });
        }
        if (taskResult.status === "fulfilled") {
          setTasks(taskResult.value.tasks);
          if (taskResult.value.warnings?.length) {
            setNotice({
              kind: "info",
              message: `Some unavailable task records were omitted: ${taskResult.value.warnings.join(" ")}`,
            });
          }
          const firstTask = taskResult.value.tasks.find(
            (task) => task.task_name && !task.remote_only,
          );
          setSelectedTaskName(firstTask?.task_name ?? "");
        } else {
          setNotice({ kind: "error", message: getErrorMessage(taskResult.reason) });
        }
        if (cropResult.status === "fulfilled") {
          setCrops(cropResult.value.crops);
          setSelectedCropIds(new Set(cropResult.value.crops.map((crop) => crop.id)));
          setSuitabilityAvailable(Boolean(cropResult.value.suitability_available));
        } else {
          setNotice({ kind: "error", message: getErrorMessage(cropResult.reason) });
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const query = selectedTaskName
      ? `?task_name=${encodeURIComponent(selectedTaskName)}`
      : "";
    let active = true;
    requestJson<EssentialCatalogResponse>(`/api/essential-layers${query}`)
      .then((result) => {
        if (active) {
          setEssentialLayers(result.layers);
          setSuitabilityInputs(result.suitability_inputs);
          setEssentialLayersError(null);
        }
      })
      .catch((error: Error) => {
        if (active) {
          setEssentialLayersError(getErrorMessage(error));
          setNotice({
            kind: "error",
            message: error.message === "Not Found"
              ? "The API server is outdated. Restart this project's backend to enable the curated layer catalog."
              : error.message,
          });
        }
      });
    return () => {
      active = false;
    };
  }, [selectedTaskName]);

  useEffect(() => {
    if (!selectedTaskName) {
      setRasters([]);
      setVisibleRasters(new Set());
      setLayerOrder([]);
      setInspection(null);
      setRastersLoading(false);
      return;
    }
    const selectedTask = tasks.find((task) => task.task_name === selectedTaskName);
    if (selectedTask?.preprocessing_status !== "completed") {
      setRasters([]);
      setVisibleRasters(new Set());
      setLayerOrder([]);
      setInspection(null);
      setRastersLoading(false);
      return;
    }
    let active = true;
    setRasters([]);
    setVisibleRasters(new Set());
    setInspection(null);
    setRastersLoading(true);
    refreshRasters(selectedTaskName, activeAnalysisName)
      .then((layers) => {
        if (active) {
          setRasters(layers);
          setLayerOrder(layers.map((layer) => layer.id));
        }
      })
      .catch((error: Error) => {
        if (active) {
          setNotice({ kind: "error", message: error.message });
        }
      })
      .finally(() => {
        if (active) setRastersLoading(false);
      });
    return () => {
      active = false;
      setTileStatus({ loading: false, error: null });
    };
  }, [selectedTaskName, activeAnalysisName, refreshRasters, tasks]);

  useEffect(() => {
    if (!aoi || !analysisCrs || resolution <= 0) {
      setAoiStats(null);
      return;
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => {
      requestJson<AoiStats>("/api/aoi/statistics", {
        method: "POST",
        body: JSON.stringify({
          name: aoiName,
          geojson: aoi,
          analysis_crs: analysisCrs,
          resolution_m: resolution,
        }),
        signal: controller.signal,
      })
        .then((stats) => {
          setAoiStats(stats);
          setAoiError(null);
        })
        .catch((error: Error) => {
          if (error.name !== "AbortError") {
            setAoiStats(null);
            setAoiError(error.message);
          }
        });
    }, 250);
    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [aoi, aoiName, analysisCrs, resolution]);

  const products = useMemo<Product[]>(() => {
    const productMap = new Map<string, Product>();
    for (const logicalLayer of essentialLayers) {
      for (const variable of logicalLayer.variables) {
        if (
          !variable.requestable_from_saved_metadata ||
          !variable.product_id ||
          !variable.layer_id
        ) {
          continue;
        }
        let product = productMap.get(variable.product_id);
        if (!product) {
          product = {
            product_id: variable.product_id,
            name: logicalLayer.source,
            spatial_resolution: variable.spatial_resolution,
            temporal_resolution: variable.temporal_resolution,
            start_date: variable.temporal_start,
            end_date: variable.temporal_end,
            layers: [],
          };
          productMap.set(product.product_id, product);
        }
        product.layers.push({
          layer_id: variable.layer_id,
          name: variable.name,
          category: logicalLayer.name,
          description: logicalLayer.description,
          units: variable.units,
          spatial_resolution: variable.spatial_resolution,
          temporal_resolution: variable.temporal_resolution,
          available: true,
          is_quality: false,
          quality_layers: [],
        });
      }
    }
    return [...productMap.values()];
  }, [essentialLayers]);
  const chosenProducts = useMemo(() => {
    const selected = new Map<string, string[]>();
    for (const product of products) {
      const layers = product.layers
        .filter((layer) => selectedLayers.has(layerKey(product.product_id, layer.layer_id)))
        .map((layer) => layer.layer_id);
      if (layers.length) selected.set(product.product_id, layers);
    }
    return selected;
  }, [products, selectedLayers]);
  const allRequestableLayerKeys = useMemo(
    () => products.flatMap((product) =>
      product.layers.map((layer) => layerKey(product.product_id, layer.layer_id)),
    ),
    [products],
  );
  const taskSelectableLayers = essentialLayers.filter(
    (logicalLayer) =>
      logicalLayer.requestable_from_saved_metadata ||
      ["soil_properties", "air_temperature", "precipitation"].includes(logicalLayer.id),
  );
  const allAcquisitionOptionsSelected =
    allRequestableLayerKeys.every((key) => selectedLayers.has(key)) &&
    ACQUISITION_PROVIDER_IDS.every((provider) => selectedProviders.has(provider));

  const visibleRasterRecords = useMemo(
    () => rasters
      .filter((layer) => visibleRasters.has(layer.id))
      .sort((left, right) => {
        const leftIndex = layerOrder.indexOf(left.id);
        const rightIndex = layerOrder.indexOf(right.id);
        return (leftIndex < 0 ? Number.MAX_SAFE_INTEGER : leftIndex)
          - (rightIndex < 0 ? Number.MAX_SAFE_INTEGER : rightIndex);
      }),
    [rasters, visibleRasters, layerOrder],
  );
  const selectedTask = tasks.find((task) => task.task_name === selectedTaskName) ?? null;
  const activeGaezScenario = gaezResult?.scenarios?.find(
    (scenario) => scenario.water_regime === gaezResultRegime,
  );
  const gaezRankedCrops = activeGaezScenario?.ranked_crops ?? [];
  const taskAcquisitionDetails = objectRecord(taskDetails?.acquisition);
  const taskProviderDetails = objectRecord(taskAcquisitionDetails?.providers);
  const suitabilityRasterRecords = rasters.filter(
    (layer) => layer.product_id === "crop-suitability",
  );
  const bestCropRaster = suitabilityRasterRecords.find(
    (layer) => layer.variable === "best_crop",
  );
  const confidenceRaster = suitabilityRasterRecords.find(
    (layer) => layer.variable === "confidence",
  );
  const scoreRasters = suitabilityRasterRecords.filter(
    (layer) => layer.variable !== "best_crop" && layer.variable !== "confidence",
  );
  const selectedProductRecords = products.filter((product) =>
    chosenProducts.has(product.product_id),
  );
  const dateOutOfRange = selectedProductRecords.some((product) => {
    if (product.temporal_resolution?.toLowerCase() === "static") return false;
    const first = product.start_date?.slice(0, 10);
    const last = product.end_date?.slice(0, 10);
    return (first && startDate < first) || (last && last.toLowerCase() !== "present" && endDate > last);
  });

  function toggleLayer(productId: string, layerId: string) {
    const key = layerKey(productId, layerId);
    setSelectedLayers((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function toggleAllAcquisitionOptions() {
    if (allAcquisitionOptionsSelected) {
      setSelectedLayers(new Set());
      setSelectedProviders(new Set());
      return;
    }
    setSelectedLayers(new Set(allRequestableLayerKeys));
    setSelectedProviders(new Set(ACQUISITION_PROVIDER_IDS));
  }

  function moveLayer(layerId: string, direction: -1 | 1) {
    setLayerOrder((current) => {
      const currentIndex = current.indexOf(layerId);
      const nextIndex = currentIndex + direction;
      if (currentIndex < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
      const next = [...current];
      [next[currentIndex], next[nextIndex]] = [next[nextIndex], next[currentIndex]];
      return next;
    });
  }

  function showSuitabilityMode(mode: "best-crop" | "individual-score") {
    setSuitabilityMode(mode);
    const selectedScore = scoreRasters.find(
      (layer) => layer.variable === selectedScoreCropId,
    ) ?? scoreRasters[0];
    const activeRaster = mode === "best-crop" ? bestCropRaster : selectedScore;
    if (!activeRaster) return;
    setVisibleRasters((current) => {
      const next = new Set(
        [...current].filter((id) => !suitabilityRasterRecords.some((layer) => layer.id === id)),
      );
      next.add(activeRaster.id);
      if (confidenceRaster) next.add(confidenceRaster.id);
      return next;
    });
    setActiveTab("Map");
  }

  async function updateAoi(geometry: GeoJson | null) {
    setAoi(geometry);
    setMapFocusAoi(null);
    setAoiStats(null);
    setAoiError(null);
    setInspection(null);
  }

  async function refreshLocalTasks() {
    try {
      await refreshTasks();
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    }
  }

  function clearArea() {
    setAoi(null);
    setAoiStats(null);
    setAoiError(null);
    setClearAoiVersion((version) => version + 1);
  }

  async function submitRequest() {
    if (!aoi || !aoiStats || (!chosenProducts.size && !selectedProviders.size)) return;
    if (aoiStats.grid_limit_exceeded) {
      setNotice({
        kind: "error",
        message: "Estimated grid exceeds the configured cell limit. Increase resolution or reduce the area.",
      });
      return;
    }
    if (endDate < startDate) {
      setNotice({ kind: "error", message: "End date must be on or after start date." });
      return;
    }
    if (dateOutOfRange) {
      setNotice({
        kind: "error",
        message: "The selected dates fall outside one or more products' temporal availability.",
      });
      return;
    }
    const requestSummary = [
      ...Array.from(chosenProducts, ([productId, layers]) =>
        `${productId}: ${layers.join(", ")}`,
      ),
      ...Array.from(selectedProviders, (provider) =>
        provider === "soil_properties"
          ? `ISRIC SoilGrids: all soil properties, ${soilDepth}`
          : provider === "nasa_power_temperature"
            ? "NASA POWER: daily mean/minimum/maximum 2 m air temperature"
            : provider === "nasa_power_precipitation"
              ? "NASA POWER: corrected precipitation"
              : "FAO/IIASA GAEZ v5: 5 historical crop suitability indices and classes, rain-fed and irrigated, high-input management",
      ),
    ].join("\n");
    const confirmed = window.confirm(
      `Create one task for this AOI and acquire the selected real datasets?\n\nArea: ${aoiName}\nArea size: ${aoiStats.area_km2.toFixed(2)} km²\nDates: ${startDate} – ${endDate}\nAnalysis grid: ${resolution} m · ${analysisCrs}\n\n${requestSummary}\n\nProvider operations will be tracked inside this task.`,
    );
    if (!confirmed) return;

    setBusy("submit");
    setNotice(null);
    try {
      const created = await requestJson<{
        task_id: string;
        task_name: string;
        acquisition_status: string;
        aoi: StoredAoi;
      }>("/api/aoi-tasks", {
        method: "POST",
        body: JSON.stringify({
          aoi_name: aoiName,
          task_name: taskName.trim() || null,
          aoi_geojson: aoi,
          start_date: startDate,
          end_date: endDate,
          analysis_crs: analysisCrs,
          resolution_m: resolution,
          soil_properties: selectedProviders.has("soil_properties"),
          soil_depth: soilDepth,
          nasa_power_temperature: selectedProviders.has("nasa_power_temperature"),
          nasa_power_precipitation: selectedProviders.has("nasa_power_precipitation"),
          gaez_v5: selectedProviders.has("gaez_v5"),
          products: Array.from(chosenProducts, ([product_id, layer_ids]) => ({
            product_id,
            layer_ids,
          })),
          confirm_acquisition: true,
        }),
      });
      if (
        created.aoi.crs !== "EPSG:4326" ||
        !sameAoiGeometry(aoi, created.aoi.geojson)
      ) {
        throw new Error("The API returned an AOI that does not match the selected geometry.");
      }
      await refreshTasks();
      setSelectedTaskName(created.task_name);
      setActiveAnalysisName(null);
      setActiveTab("Map");
      setNotice({
        kind: "info",
        message: `Task ${created.task_name} created. Acquiring selected providers…`,
      });

      let acquisitionStatus = created.acquisition_status;
      let latestTaskSummary: Task | undefined;
      let acquisitionStage = "";
      for (let attempt = 0; attempt < 180 && ["queued", "running"].includes(acquisitionStatus); attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        const latest = await requestJson<{ tasks: Task[] }>("/api/tasks");
        setTasks(latest.tasks);
        latestTaskSummary = latest.tasks.find((task) => task.task_id === created.task_id);
        acquisitionStatus = latestTaskSummary?.acquisition?.status ?? "failed";
        acquisitionStage = latestTaskSummary?.acquisition?.stage ?? "";
        if (acquisitionStage === "awaiting_provider_completion") break;
        const runningProviders = Object.entries(latestTaskSummary?.acquisition?.providers ?? {})
          .filter(([, provider]) => ["queued", "running"].includes(provider.status))
          .map(([name]) => name.replaceAll("_", " "));
        if (["queued", "running"].includes(acquisitionStatus)) {
          setNotice({
            kind: "info",
            message: runningProviders.length
              ? `Acquiring ${runningProviders.join(", ")} for ${created.task_name}…`
              : `Waiting for providers to start for ${created.task_name}…`,
          });
        }
      }

      const [layers, catalog] = await Promise.all([
        refreshRasters(created.task_name, null),
        requestJson<EssentialCatalogResponse>(
          `/api/essential-layers?task_name=${encodeURIComponent(created.task_name)}`,
        ),
      ]);
      setRasters(layers);
      setVisibleRasters(new Set());
      setLayerOrder(layers.map((layer) => layer.id));
      setEssentialLayers(catalog.layers);
      setSuitabilityInputs(catalog.suitability_inputs);
      const finalTasks = await requestJson<{ tasks: Task[] }>("/api/tasks");
      setTasks(finalTasks.tasks);
      latestTaskSummary = finalTasks.tasks.find((task) => task.task_id === created.task_id);
      const providerFailures = Object.entries(latestTaskSummary?.acquisition?.providers ?? {})
        .filter(([, provider]) => ["failed", "partial"].includes(provider.status))
        .map(([name, provider]) => `${name}: ${provider.error ?? "failed"}`);
      setNotice({
        kind: providerFailures.length ? "error" : "success",
        message: providerFailures.length
          ? `Acquisition finished with provider errors: ${providerFailures.join("; ")}. Successful datasets remain available on ${created.task_name}.`
          : acquisitionStatus === "running" || acquisitionStatus === "queued"
            ? `Acquisition is still running for ${created.task_name}. The task list shows provider progress.`
            : `Acquisition completed for ${created.task_name}. Enable acquired layers in the map layer selector.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function searchLocation(event: FormEvent) {
    event.preventDefault();
    const value = search.trim();
    if (!value) return;
    const coordinates = value.match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
    if (coordinates) {
      const lat = Number(coordinates[1]);
      const lon = Number(coordinates[2]);
      if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
        setSearchTarget({ lat, lon });
        return;
      }
    }
    setBusy("search");
    try {
      const url = new URL("https://nominatim.openstreetmap.org/search");
      url.searchParams.set("format", "jsonv2");
      url.searchParams.set("limit", "1");
      url.searchParams.set("q", value);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Location search failed (${response.status}).`);
      const matches = (await response.json()) as Array<{
        lat: string;
        lon: string;
        display_name: string;
      }>;
      if (!matches.length) throw new Error("No matching location found.");
      setSearchTarget({ lat: Number(matches[0].lat), lon: Number(matches[0].lon) });
      setNotice({ kind: "info", message: matches[0].display_name });
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function inspectPoint(point: { lat: number; lon: number }) {
    if (!selectedTaskName) return;
    const layer = rasters.find((item) => visibleRasters.has(item.id));
    if (!layer) {
      setInspection({
        lat: point.lat.toFixed(5),
        lon: point.lon.toFixed(5),
        message: "Enable a processed raster layer to inspect its value.",
      });
      return;
    }
    try {
      const query = new URLSearchParams({
        task_name: selectedTaskName,
        lat: String(point.lat),
        lon: String(point.lon),
      });
      const value = layer.product_id === "crop-suitability"
        ? await requestJson<Record<string, unknown>>(
            `/api/suitability/cell?${new URLSearchParams({
              task_name: selectedTaskName,
              lat: String(point.lat),
              lon: String(point.lon),
              ...(layer.analysis_name ? { analysis_name: layer.analysis_name } : {}),
            })}`,
          )
        : await requestJson<Record<string, unknown>>(
            `/api/layers/${layer.id}/value?${query}`,
          );
      setInspection(value);
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    }
  }

  async function monitorTask(taskId: string | null) {
    if (!taskId) return;
    setBusy(taskId);
    setNotice(null);
    try {
      const state = await requestJson<{
        task_id: string;
        task_status: string;
        remote_only?: boolean;
      }>(`/api/tasks/${taskId}/monitor`, { method: "POST" });
      if (state.remote_only) {
        setTasks((current) =>
          current.map((task) =>
            task.task_id === taskId
              ? { ...task, task_status: state.task_status }
              : task,
          ),
        );
      } else {
        await refreshTasks();
      }
      setNotice({ kind: "success", message: `Status refreshed for ${taskId}.` });
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function showTaskDetails(taskId: string | null) {
    if (!taskId) return;
    if (taskDetails?.task_id === taskId) {
      setTaskDetails(null);
      return;
    }
    setBusy(taskId);
    try {
      setTaskDetails(await requestJson<Record<string, unknown>>(`/api/tasks/${taskId}`));
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function downloadTask(taskId: string | null) {
    if (!taskId) return;
    setBusy(taskId);
    setNotice(null);
    try {
      const manifest = await requestJson<Task>(`/api/tasks/${taskId}/download`, {
        method: "POST",
      });
      await refreshTasks();
      setNotice({
        kind: "success",
        message: `Download state: ${manifest.download_status}.`,
      });
      await refreshTasks();
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function downloadEverything(task: Task) {
    if (!task.task_id || !task.task_name) return;
    setBusy(task.task_id);
    setNotice(null);
    try {
      const prepared = await requestJson<{
        status: "ready" | "partial" | "pending";
        linked_task_count: number;
        integrated_raster_count: number;
        newly_integrated_raster_count: number;
        failures: Array<{ task_name: string; error: string }>;
      }>(`/api/aoi-tasks/${encodeURIComponent(task.task_id)}/download`, {
        method: "POST",
      });
      const [layers, catalog] = await Promise.all([
        refreshRasters(task.task_name, null),
        requestJson<EssentialCatalogResponse>(
          `/api/essential-layers?task_name=${encodeURIComponent(task.task_name)}`,
        ),
      ]);
      setRasters(layers);
      setLayerOrder(layers.map((layer) => layer.id));
      setEssentialLayers(catalog.layers);
      setSuitabilityInputs(catalog.suitability_inputs);
      setSelectedTaskName(task.task_name);
      setActiveAnalysisName(null);
      await refreshTasks();
      const failureDetails = prepared.failures
        .map((failure) => `${failure.task_name}: ${failure.error}`)
        .join("; ");
      setNotice({
        kind: prepared.status === "pending" ? "info" : prepared.status === "partial" ? "error" : "success",
        message: prepared.status === "pending"
          ? "Some AppEEARS operations are still running. The unified task and any completed provider outputs have been preserved; prepare the data again when those operations finish."
          : prepared.status === "partial"
          ? `Some linked datasets could not be prepared. ${failureDetails} Successfully integrated layers remain on the analysis server.`
          : `All data is stored on the analysis server and ready for ${task.task_name}: ${prepared.integrated_raster_count} raster layers (${prepared.newly_integrated_raster_count} newly added) from ${prepared.linked_task_count} linked NASA tasks.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setBusy(null);
    }
  }

  async function retryAoiTask(task: Task) {
    if (!task.task_id) return;
    setBusy(task.task_id);
    setNotice(null);
    try {
      const result = await requestJson<{
        task_name: string;
        retried_providers: string[];
      }>(`/api/aoi-tasks/${encodeURIComponent(task.task_id)}/retry`, {
        method: "POST",
      });
      await refreshTasks();
      setNotice({
        kind: "info",
        message: `Retrying failed providers for ${result.task_name}: ${result.retried_providers.join(", ")}.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setBusy(null);
    }
  }

  async function standardizeTask(taskId: string) {
    setBusy(taskId);
    setNotice(null);
    try {
      const result = await requestJson<{ task_name: string; raster_count: number }>(
        `/api/tasks/${taskId}/standardize`,
        { method: "POST" },
      );
      await refreshTasks();
      setSelectedTaskName(result.task_name);
      setActiveAnalysisName(null);
      setNotice({
        kind: "success",
        message: `Standardized ${result.raster_count} raster(s) for ${result.task_name}.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function loadClimateData() {
    const task = tasks.find((item) => item.task_name === selectedTaskName);
    if (!task?.task_id) {
      setNotice({ kind: "error", message: "Select a registered NASA task before fetching climate data." });
      return;
    }
    const confirmed = window.confirm(
      `Fetch daily NASA POWER air temperature and precipitation for ${task.task_name} (${task.start_date ?? "task dates"} to ${task.end_date ?? "task dates"})?`,
    );
    if (!confirmed) return;
    setBusy("climate");
    setNotice(null);
    try {
      const result = await requestJson<{
        new_variables: string[];
        already_integrated: string[];
      }>(`/api/tasks/${encodeURIComponent(task.task_id)}/climate`, {
        method: "POST",
      });
      const [layers, catalog] = await Promise.all([
        refreshRasters(task.task_name!, null),
        requestJson<EssentialCatalogResponse>(
          `/api/essential-layers?task_name=${encodeURIComponent(task.task_name!)}`,
        ),
      ]);
      setRasters(layers);
      setLayerOrder(layers.map((layer) => layer.id));
      setEssentialLayers(catalog.layers);
      setSuitabilityInputs(catalog.suitability_inputs);
      setNotice({
        kind: "success",
        message: result.new_variables.length
          ? `Fetched NASA POWER ${result.new_variables.join(", ")} for ${task.task_name}. Layers are available but remain hidden on the map.`
          : `NASA POWER climate layers are already available for ${task.task_name}.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setBusy(null);
    }
  }

  async function loadSoilData() {
    const task = tasks.find((item) => item.task_name === selectedTaskName);
    if (!task?.task_id) {
      setNotice({ kind: "error", message: "Select a registered NASA task before fetching soil data." });
      return;
    }
    setBusy("soil");
    setNotice(null);
    try {
      const result = await requestJson<{
        new_variables: string[];
        already_integrated: string[];
      }>(
        `/api/tasks/${encodeURIComponent(task.task_id)}/soil?depth=${encodeURIComponent(soilDepth)}`,
        { method: "POST" },
      );
      const [layers, catalog] = await Promise.all([
        refreshRasters(task.task_name!, null),
        requestJson<EssentialCatalogResponse>(
          `/api/essential-layers?task_name=${encodeURIComponent(task.task_name!)}`,
        ),
      ]);
      setRasters(layers);
      setLayerOrder(layers.map((layer) => layer.id));
      setEssentialLayers(catalog.layers);
      setSuitabilityInputs(catalog.suitability_inputs);
      setNotice({
        kind: "success",
        message: result.new_variables.length
          ? `Fetched SoilGrids ${result.new_variables.join(", ")} at ${soilDepth} for ${task.task_name}. Layers are available but remain hidden on the map.`
          : `SoilGrids data at ${soilDepth} is already available for ${task.task_name}.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setBusy(null);
    }
  }

  async function runSuitabilityAnalysis() {
    if (!selectedTaskName) return;
    setBusy("suitability");
    setNotice(null);
    try {
      const result = await requestJson<{
        task_name: string;
        task_id: string;
        analysis_name: string;
        input_raster_count: number;
        missing_variables: string[];
        cells_with_data: number;
      }>("/api/suitability/analyze", {
        method: "POST",
        body: JSON.stringify({
          task_name: selectedTaskName,
          analysis_name: analysisName.trim() || null,
          crop_ids: selectedCropIds.size ? Array.from(selectedCropIds) : [],
        }),
      });
      setActiveAnalysisName(result.analysis_name);
      await refreshTasks();
      setNotice({
        kind: "success",
        message: `${result.analysis_name}: ${result.input_raster_count} input raster(s), ${result.cells_with_data} cells with data. Missing: ${result.missing_variables.join(", ") || "none"}.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function runGaezAnalysis() {
    if (!selectedTask?.task_id || !gaezCropCodes.size) return;
    setBusy("gaez");
    setNotice(null);
    try {
      const request = await requestJson<{ run_id: string; status: string }>(
        `/api/aoi-tasks/${encodeURIComponent(selectedTask.task_id)}/gaez-suitability`,
        {
          method: "POST",
          body: JSON.stringify({
            crop_codes: Array.from(gaezCropCodes),
            water_regime: gaezWaterRegime,
            management_level: gaezManagementLevel,
          }),
        },
      );
      let result = await requestJson<GaezResult>(
        `/api/aoi-tasks/${encodeURIComponent(selectedTask.task_id)}/gaez-suitability?run_id=${encodeURIComponent(request.run_id)}`,
      );
      setGaezResult(result);
      for (let attempt = 0; attempt < 240 && ["queued", "running"].includes(result.status); attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
        result = await requestJson<GaezResult>(
          `/api/aoi-tasks/${encodeURIComponent(selectedTask.task_id)}/gaez-suitability?run_id=${encodeURIComponent(request.run_id)}`,
        );
        setGaezResult(result);
      }
      if (["queued", "running"].includes(result.status)) {
        throw new Error("GAEZ is still processing. Check the task details to follow the provider status.");
      }
      if (result.status === "failed") {
        throw new Error(result.error ?? "GAEZ could not produce a suitability result.");
      }
      const layers = await refreshRasters(selectedTaskName, activeAnalysisName);
      setRasters(layers);
      setLayerOrder(layers.map((layer) => layer.id));
      setNotice({
        kind: result.status === "partial" ? "info" : "success",
        message: result.status === "partial"
          ? "GAEZ returned partial results. Failed crop operations and coverage exclusions are shown below."
          : `GAEZ v5 ${result.period} suitability is ready for ${selectedTaskName}.`,
      });
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setBusy(null);
    }
  }

  async function showGaezMap(crop: GaezCropResult) {
    const classLayer = crop.class_map_layer
      ? rasters.find(
        (layer) => layer.product_id === "FAO_GAEZ_V5"
          && layer.variable === crop.class_map_layer,
      )
      : undefined;
    const indexLayer = rasters.find(
      (layer) => layer.product_id === "FAO_GAEZ_V5"
        && layer.variable === crop.continuous_map_layer,
    );
    const activeLayer = classLayer ?? indexLayer;
    if (!activeLayer) {
      setNotice({
        kind: "info",
        message: `No GAEZ map layer is available for ${crop.crop_name} in this scenario.`,
      });
      return;
    }
    setVisibleRasters((current) => {
      const next = new Set(
        [...current].filter(
          (id) => !rasters.some((layer) => layer.id === id && layer.product_id === "FAO_GAEZ_V5"),
        ),
      );
      next.add(activeLayer.id);
      return next;
    });
    setActiveTab("Map");
    if (!gaezResult?.task_id) {
      setNotice({
        kind: "error",
        message: "The GAEZ result has no task identifier, so its AOI cannot be located on the map.",
      });
      return;
    }
    try {
      const task = await requestJson<{ aoi?: { geojson?: GeoJson } }>(
        `/api/tasks/${encodeURIComponent(gaezResult.task_id)}`,
      );
      if (!task.aoi?.geojson) {
        setNotice({
          kind: "info",
          message: "The map layer is selected, but the task AOI geometry is unavailable for zooming.",
        });
        return;
      }
      setMapFocusAoi(task.aoi.geojson);
      setNotice({
        kind: "info",
        message: classLayer
          ? `Showing official suitability classes for ${crop.crop_name} within the task AOI.`
          : `Official class data is unavailable for ${crop.crop_name}; showing its continuous GAEZ index within the task AOI.`,
      });
    } catch (error) {
      setNotice({
        kind: "error",
        message: `The map layer is selected, but the task AOI could not be loaded: ${getErrorMessage(error)}`,
      });
    }
  }

  function selectTask(taskName: string) {
    setRasters([]);
    setVisibleRasters(new Set());
    setLayerOrder([]);
    setInspection(null);
    setTaskDetails(null);
    setMapFocusAoi(null);
    setActiveAnalysisName(null);
    setSelectedTaskName(taskName);
  }

  async function syncRemoteTasks() {
    setBusy("sync");
    try {
      await refreshTasks(true);
      setNotice({ kind: "success", message: "NASA task list synchronized." });
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message });
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <div className="brand-name">FIELD ATLAS</div>
            <div className="brand-caption">EARTH OBSERVATION · UZBEKISTAN</div>
          </div>
        </div>
        <nav className="top-nav" aria-label="Main navigation">
          {["Map", "Data", "Tasks", "Crop suitability"].map((item) => (
            <button
              className={activeTab === item ? "nav-item active" : "nav-item"}
              key={item}
              onClick={() => {
                setActiveTab(item);
                if (item === "Data") {
                  document.getElementById("datasets-panel")?.scrollIntoView({ behavior: "smooth" });
                } else if (item === "Tasks") {
                  document.getElementById("tasks-panel")?.scrollIntoView({ behavior: "smooth" });
                }
              }}
              type="button"
            >
              {item}
              {item === "Tasks" && tasks.length > 0 && (
                <span className="nav-count">{tasks.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="header-status">
          <span className="status-dot" />
          <span>LOCAL WORKSPACE</span>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <section className="sidebar-intro">
            <div className="eyebrow">ANALYSIS WORKSPACE</div>
            <h1>Land &amp; crop intelligence</h1>
            <p>Draw an area, choose environmental data, and track acquisition for a new task.</p>
          </section>

          <section className="panel-section area-section">
            <div className="section-heading">
              <span className="section-index">01</span>
              <div>
                <h2>Area of interest</h2>
                <p>Draw or edit a polygon on the map</p>
              </div>
            </div>
            <form className="search-form" onSubmit={searchLocation}>
              <span className="search-icon" aria-hidden="true">⌕</span>
              <input
                aria-label="Search a place or coordinates"
                placeholder="Search place or lat, lon"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
              <button type="submit" disabled={busy === "search"} aria-label="Search">
                ↗
              </button>
            </form>
            <div className="map-tip">
              <span className="tip-symbol">↗</span>
              Use Draw rectangle on the map: click two opposite corners or drag across the area. Polygon: click vertices, then double-click to finish.
            </div>
            {aoiStats ? (
              <div className="aoi-summary">
                <div className="summary-title">
                  <span className="summary-dot" />
                  <input
                    className="aoi-name-input"
                    aria-label="Area of interest name"
                    value={aoiName}
                    onChange={(event) => setAoiName(event.target.value)}
                  />
                  <button type="button" className="text-button" onClick={clearArea}>
                    Clear
                  </button>
                </div>
                <div className="aoi-grid">
                  <span>AREA</span><strong>{aoiStats.area_km2.toFixed(2)} km²</strong>
                  <span>TYPE</span><strong>{aoiStats.geometry_type}</strong>
                  <span>GRID ESTIMATE</span>
                  <strong className={aoiStats.grid_limit_exceeded ? "danger-text" : ""}>
                    {aoiStats.estimated_grid_cells.toLocaleString()} cells
                  </strong>
                  <span>ANALYSIS CRS</span><strong>{aoiStats.analysis_crs}</strong>
                </div>
                <div className="bounds">
                  {aoiStats.bounds.south.toFixed(4)}° – {aoiStats.bounds.north.toFixed(4)}° N
                  <br />
                  {aoiStats.bounds.west.toFixed(4)}° – {aoiStats.bounds.east.toFixed(4)}° E
                </div>
              </div>
            ) : (
              <div className="empty-aoi">
                {aoiError ?? (aoi
                  ? "Calculating selected area bounds…"
                  : "Draw an area to see bounds and grid estimates.")}
              </div>
            )}
          </section>

          <section className="panel-section dataset-section" id="datasets-panel">
            <div className="section-heading dataset-heading">
              <span className="section-index">02</span>
              <div>
                <h2>Essential data</h2>
                <p>{taskSelectableLayers.length} selectable groups · select data for a new task</p>
              </div>
              <button
                type="button"
                className="select-all-button"
                disabled={loading || taskSelectableLayers.length === 0}
                onClick={toggleAllAcquisitionOptions}
              >
                {allAcquisitionOptionsSelected ? "Clear selection" : "Select all available"}
              </button>
            </div>
            {loading ? (
              <div className="loading-row">Loading the curated layer catalog…</div>
            ) : essentialLayers.length === 0 ? (
              <div className="empty-state">
                {essentialLayersError === "Not Found"
                  ? "This API server does not include the curated layer catalog. Restart the project API server so /api/essential-layers is available."
                  : essentialLayersError
                    ? `Could not load the curated layer catalog: ${essentialLayersError}`
                    : "No curated layers are available from the API."}
              </div>
            ) : (
              <div className="essential-collection-list">
                {taskSelectableLayers.map((logicalLayer) => (
                  <details className="essential-collection-item" key={logicalLayer.id}>
                    <summary>
                      <span className="essential-summary-name">{logicalLayer.name}</span>
                      <span className={`availability-pill ${logicalLayer.processed_on_selected_task ? "is-processed" : logicalLayer.requestable_from_saved_metadata ? "is-requestable" : "is-unavailable"}`}>
                        {logicalLayer.processed_on_selected_task
                        ? "Processed task data"
                          : logicalLayer.requestable_from_saved_metadata
                          ? "NASA catalog metadata"
                          : logicalLayer.availability_status === "available_on_demand"
                            ? "Available on demand"
                          : logicalLayer.availability_status === "metadata_not_verified"
                            ? "Metadata not verified"
                            : logicalLayer.availability_status === "derived_when_source_is_processed"
                              ? "Derived from task DEM"
                            : "Not integrated"}
                      </span>
                    </summary>
                    <p className="essential-description">{logicalLayer.description}</p>
                    <div className="essential-source">
                      {logicalLayer.category} · {logicalLayer.source}
                      {logicalLayer.required_for_suitability
                        ? " · suitability input"
                        : " · visualization/monitoring"}
                    </div>
                    {logicalLayer.id === "soil_properties" && (
                      <div className="provider-selection">
                        <label className="essential-acquire-option">
                          <input
                            type="checkbox"
                            checked={selectedProviders.has("soil_properties")}
                            onChange={(event) => setSelectedProviders((current) => {
                              const next = new Set(current);
                              if (event.target.checked) next.add("soil_properties");
                              else next.delete("soil_properties");
                              return next;
                            })}
                          />
                          <span>Acquire all soil properties for this AOI</span>
                        </label>
                        <label className="soil-depth-control">
                          <span>SOIL DEPTH</span>
                          <select
                            aria-label="SoilGrids acquisition depth"
                            value={soilDepth}
                            onChange={(event) => setSoilDepth(event.target.value)}
                          >
                            {["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"].map((depth) => (
                              <option key={depth} value={depth}>{depth}</option>
                            ))}
                          </select>
                        </label>
                      </div>
                    )}
                    {logicalLayer.id === "air_temperature" && (
                      <label className="essential-acquire-option provider-selection">
                        <input
                          type="checkbox"
                          checked={selectedProviders.has("nasa_power_temperature")}
                          onChange={(event) => setSelectedProviders((current) => {
                            const next = new Set(current);
                            if (event.target.checked) next.add("nasa_power_temperature");
                            else next.delete("nasa_power_temperature");
                            return next;
                          })}
                        />
                        <span>Acquire daily mean, minimum, and maximum 2 m temperature</span>
                      </label>
                    )}
                    {logicalLayer.id === "precipitation" && (
                      <label className="essential-acquire-option provider-selection">
                        <input
                          type="checkbox"
                          checked={selectedProviders.has("nasa_power_precipitation")}
                          onChange={(event) => setSelectedProviders((current) => {
                            const next = new Set(current);
                            if (event.target.checked) next.add("nasa_power_precipitation");
                            else next.delete("nasa_power_precipitation");
                            return next;
                          })}
                        />
                        <span>Acquire corrected precipitation totals</span>
                      </label>
                    )}
                    {logicalLayer.variables.map((variable) => {
                      const canRequest = variable.requestable_from_saved_metadata
                        && typeof variable.product_id === "string"
                        && typeof variable.layer_id === "string";
                      const productId = variable.product_id;
                      const layerId = variable.layer_id;
                      const key = canRequest
                        ? layerKey(productId ?? "", layerId ?? "")
                        : variable.id;
                      return (
                        <div className="essential-variable" key={variable.id}>
                          {canRequest ? (
                            <label className="essential-request-option">
                              <input
                                type="checkbox"
                                checked={selectedLayers.has(key)}
                                onChange={() => {
                                  if (productId && layerId) toggleLayer(productId, layerId);
                                }}
                              />
                              <span>{variable.name}</span>
                              <small>
                                {variable.spatial_resolution} · {variable.temporal_resolution}
                              </small>
                            </label>
                          ) : (
                            <div className="essential-unavailable-variable">
                              <span>{variable.name}</span>
                              <small>
                                {variable.availability_status === "available_on_demand"
                                  ? `Fetch from ${logicalLayer.source} for this task`
                                  : variable.availability_status === "derived_when_source_is_processed"
                                  ? "Derived when source elevation is processed"
                                  : variable.availability_status === "metadata_not_verified"
                                  ? "No verified product metadata"
                                  : variable.availability_detail ?? "Source not integrated"}
                              </small>
                            </div>
                          )}
                          {(canRequest || variable.units || variable.spatial_resolution || variable.temporal_resolution) && (
                            <div className="essential-metadata">
                              {variable.units || "Units not specified"}
                              {variable.spatial_resolution && ` · ${variable.spatial_resolution}`}
                              {variable.temporal_resolution && ` · ${variable.temporal_resolution}`}
                              {variable.temporal_start && ` · ${variable.temporal_start}–${variable.temporal_end ?? "present"}`}
                              {variable.scale_factor != null && ` · scale ${variable.scale_factor}`}
                              {variable.add_offset != null && variable.add_offset !== 0 && ` · offset ${variable.add_offset}`}
                              {variable.fill_values.length > 0 && ` · fill ${variable.fill_values.join("/")}`}
                              {variable.quality_layers.length > 0 && ` · QA ${variable.quality_layers.join(", ")}`}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </details>
                ))}
              </div>
            )}
            <label className="essential-acquire-option provider-selection gaez-acquire-option">
              <input
                type="checkbox"
                checked={selectedProviders.has("gaez_v5")}
                onChange={(event) => setSelectedProviders((current) => {
                  const next = new Set(current);
                  if (event.target.checked) next.add("gaez_v5");
                  else next.delete("gaez_v5");
                  return next;
                })}
              />
              <span>
                Acquire FAO/IIASA GAEZ v5 historical crop suitability
                <small>
                  5 crops · separate rain-fed/irrigated rankings · high-input management
                </small>
              </span>
            </label>
          </section>

          <section className="panel-section request-section">
            <div className="section-heading">
              <span className="section-index">03</span>
              <div>
                <h2>Collection window</h2>
                <p>Original product resolution is preserved</p>
              </div>
            </div>
            <div className="date-row">
              <label>
                <span>START DATE</span>
                <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
              </label>
              <label>
                <span>END DATE</span>
                <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
              </label>
            </div>
            <label className="task-name-field">
              <span>TASK NAME · OPTIONAL</span>
              <input
                value={taskName}
                onChange={(event) => setTaskName(event.target.value)}
                placeholder="Generated automatically if blank"
                maxLength={120}
              />
            </label>
            {dateOutOfRange && (
              <div className="validation-warning">
                Selected dates are outside the availability window of at least one product.
              </div>
            )}
            <div className="spatial-row">
              <label>
                <span>ANALYSIS CRS</span>
                <input value={analysisCrs} onChange={(event) => setAnalysisCrs(event.target.value)} />
              </label>
              <label>
                <span>GRID SIZE · METERS</span>
                <input
                  type="number"
                  min="1"
                  max="100000"
                  list="resolution-options"
                  value={resolution}
                  onChange={(event) => setResolution(Number(event.target.value))}
                />
                <datalist id="resolution-options">
                  <option value="250" />
                  <option value="500" />
                  <option value="1000" />
                </datalist>
              </label>
            </div>
            <p className="resolution-note">
              Grid size affects analysis only; it does not increase source-data detail.
            </p>
            <button
              type="button"
              className="primary-button"
              disabled={
                !aoi ||
                !aoiStats ||
                Boolean(aoiError) ||
                (!chosenProducts.size && !selectedProviders.size) ||
                dateOutOfRange ||
                endDate < startDate ||
                aoiStats.grid_limit_exceeded ||
                Boolean(busy)
              }
              onClick={submitRequest}
            >
              {busy === "submit" ? "Creating task & acquiring data…" : "Create task & acquire selected data"}
              <span>→</span>
            </button>
            <div className="selection-count">
              {selectedLayers.size} AppEEARS layer{selectedLayers.size === 1 ? "" : "s"} · {selectedProviders.size} external dataset{selectedProviders.size === 1 ? "" : "s"}
            </div>
            <p className="resolution-note">
              Map visibility is independent of suitability inputs. {suitabilityInputs?.status} NASA catalog metadata does not guarantee downloaded data or full AOI coverage.
            </p>
          </section>
        </aside>

        <section
          className={`map-area ${activeTab === "Tasks" ? "map-tasks-visible" : ""}`}
          aria-label="Map workspace"
        >
          <div className="map-toolbar">
            <div className="map-title">
              <span className="map-title-mark">◎</span>
              <div>
                <strong>Global AOI</strong>
                <span>Interactive analysis map</span>
              </div>
            </div>
            <div className="map-badges">
              <label className="map-task-select">
                <span>DATA TASK</span>
                <select
                  aria-label="Selected data task for map layers"
                  value={selectedTaskName}
                  onChange={(event) => selectTask(event.target.value)}
                >
                  <option value="">Select task</option>
                  {tasks.filter((task) => task.task_name && !task.remote_only).map((task) => (
                    <option key={task.task_name} value={task.task_name!}>
                      {task.task_name}
                    </option>
                  ))}
                </select>
              </label>
              <span className="map-badge"><i /> {aoi ? "AREA SELECTED" : "DRAW AN AREA"}</span>
              <span className="map-badge muted">{analysisCrs || "CRS not set"}</span>
            </div>
          </div>
          <MapView
            focusAoi={mapFocusAoi}
            clearRequest={clearAoiVersion}
            rasterLayers={visibleRasterRecords}
            opacity={opacity}
            onRasterStatusChange={setTileStatus}
            searchTarget={searchTarget}
            onAoiChange={updateAoi}
            onMapClick={inspectPoint}
          />
          <div className="map-attribution-note">NASA AppEEARS · OpenStreetMap</div>

          <aside className="map-card layer-control-card">
            <div className="map-card-heading">
              <div>
                <strong>Essential map layers</strong>
                <span>{rastersLoading ? "Loading task layer metadata…" : selectedTaskName || "Select a task"}</span>
              </div>
              <span className="layers-icon">▤</span>
            </div>
            {!selectedTaskName && (
              <div className="map-data-status">
                Draw an AOI and use the panel to create a task and acquire the selected datasets.
              </div>
            )}
            {selectedTaskName && selectedTask?.preprocessing_status !== "completed" && (
              <div className="map-data-status">
                Download and preprocess this task before requesting task-scoped soil or climate data.
              </div>
            )}
            {selectedTaskName && selectedTask?.preprocessing_status === "completed" && (
              <>
                <button
                  type="button"
                  className="climate-load-button"
                  disabled={Boolean(busy)}
                  onClick={loadClimateData}
                >
                  {busy === "climate" ? "Fetching NASA POWER climate…" : "Fetch NASA POWER climate"}
                  <small>Air temperature + precipitation · task date range</small>
                </button>
                <label className="soil-depth-control">
                  <span>SOIL DEPTH</span>
                  <select
                    aria-label="SoilGrids depth interval"
                    value={soilDepth}
                    disabled={Boolean(busy)}
                    onChange={(event) => setSoilDepth(event.target.value)}
                  >
                    {["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"].map((depth) => (
                      <option key={depth} value={depth}>{depth}</option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="climate-load-button"
                  disabled={Boolean(busy)}
                  onClick={loadSoilData}
                >
                  {busy === "soil" ? "Fetching ISRIC SoilGrids…" : "Fetch Soil Properties"}
                  <small>pH, sand, silt, clay, organic carbon · selected depth</small>
                </button>
              </>
            )}
            {tileStatus.loading && <div className="map-data-status">Loading visible raster tiles…</div>}
            {tileStatus.error && <div className="map-data-error">{tileStatus.error}</div>}
            <div className="essential-map-list">
              {essentialLayers.map((logicalLayer) => (
                <details className="essential-map-item" key={logicalLayer.id}>
                  <summary>
                    <span>{logicalLayer.name}</span>
                    <small>
                      {logicalLayer.processed_on_selected_task
                        ? "On task"
                        : logicalLayer.requestable_from_saved_metadata
                          ? "No task data"
                          : logicalLayer.availability_status === "available_on_demand"
                            ? "Available on demand"
                          : logicalLayer.availability_status === "derived_when_source_is_processed"
                            ? "Derived from task DEM"
                          : "Unavailable"}
                    </small>
                  </summary>
                  {logicalLayer.variables.map((variable) => {
                    if (!variable.product_id || !variable.layer_id) {
                      return (
                        <div className="map-layer-unavailable" key={variable.id}>
                          {variable.name} · {logicalLayer.source}
                        </div>
                      );
                    }
                    const records = rasters.filter((raster) =>
                      raster.product_id === variable.product_id
                      && raster.variable === variable.layer_id
                    );
                    return records.length ? records.map((layer) => {
                      const stackIndex = layerOrder.indexOf(layer.id);
                      return (
                        <div className="map-layer-entry" key={layer.id}>
                          <label className="map-layer-option">
                            <input
                              type="checkbox"
                              checked={visibleRasters.has(layer.id)}
                              onChange={() =>
                                setVisibleRasters((current) => {
                                  const next = new Set(current);
                                  if (next.has(layer.id)) next.delete(layer.id);
                                  else next.add(layer.id);
                                  return next;
                                })
                              }
                            />
                            <span>{variable.name}</span>
                            <small>
                              {layer.observation_date
                                ?? (layer.temporal_start
                                  ? `${layer.temporal_start} – ${layer.temporal_end ?? ""}`
                                  : "undated")}
                            </small>
                          </label>
                          {visibleRasters.has(layer.id) && (
                            <div className="layer-order-controls" aria-label={`Order ${variable.name}`}>
                              <button
                                type="button"
                                title="Bring layer forward"
                                aria-label={`Bring ${variable.name} forward`}
                                disabled={stackIndex >= layerOrder.length - 1}
                                onClick={() => moveLayer(layer.id, 1)}
                              >↑</button>
                              <button
                                type="button"
                                title="Send layer backward"
                                aria-label={`Send ${variable.name} backward`}
                                disabled={stackIndex <= 0}
                                onClick={() => moveLayer(layer.id, -1)}
                              >↓</button>
                            </div>
                          )}
                          <details className="map-layer-metadata">
                            <summary>Source metadata</summary>
                            <span>
                              {logicalLayer.source} · {variable.spatial_resolution ?? "resolution unavailable"} · {variable.temporal_resolution ?? "temporal resolution unavailable"}
                              {variable.temporal_start && ` · ${variable.temporal_start}–${variable.temporal_end ?? "present"}`}
                            </span>
                            <span>{variable.units ?? "Units unavailable"}{variable.scale_factor != null ? ` · scale ${variable.scale_factor}` : ""}{variable.add_offset != null ? ` · offset ${variable.add_offset}` : ""}</span>
                            {variable.quality_layers.length > 0 && (
                              <span>QA dependency: {variable.quality_layers.join(", ")} (filter not applied)</span>
                            )}
                          </details>
                        </div>
                      );
                    }) : (
                      <div className="map-layer-unavailable" key={variable.id}>
                        {variable.name} · {variable.processed_on_selected_task
                          ? "No verified raster record"
                          : logicalLayer.requestable_from_saved_metadata
                            ? "Not processed for this task"
                            : logicalLayer.source}
                      </div>
                    );
                  })}
                </details>
              ))}
              {rasters.some((layer) => layer.product_id === "crop-suitability") && (
                <details className="essential-map-item">
                  <summary><span>Crop Suitability</span><small>Analysis outputs</small></summary>
                  {rasters.filter((layer) => layer.product_id === "crop-suitability").map((layer) => (
                    <label className="map-layer-option" key={layer.id}>
                      <input
                        type="checkbox"
                        checked={visibleRasters.has(layer.id)}
                        onChange={() => setVisibleRasters((current) => {
                          const next = new Set(current);
                          if (next.has(layer.id)) next.delete(layer.id);
                          else next.add(layer.id);
                          return next;
                        })}
                      />
                      <span>{layer.name}</span>
                      <small>{layer.analysis_name ?? ""}</small>
                    </label>
                  ))}
                </details>
              )}
            </div>
            {visibleRasterRecords.length > 0 && (
              <>
                {visibleRasters.size > 0 && (
                  <label className="opacity-control">
                    <span>Opacity</span>
                    <input
                      type="range"
                      min="0.2"
                      max="1"
                      step="0.05"
                      value={opacity}
                      onChange={(event) => setOpacity(Number(event.target.value))}
                    />
                    <span>{Math.round(opacity * 100)}%</span>
                  </label>
                )}
                {visibleRasterRecords.map((legendLayer) => (
                  <div className="raster-legend" key={legendLayer.id}>
                    <div className="legend-name">
                      {legendLayer.name} {legendLayer.units ? `· ${legendLayer.units}` : ""}
                    </div>
                    {legendLayer.category_labels && legendLayer.category_colors && (
                      <>
                        <div className="categorical-map-title">
                          {legendLayer.product_id === "FAO_GAEZ_V5"
                            ? "Official GAEZ suitability classes"
                            : "Map categories"}
                        </div>
                        <div className="categorical-map-legend">
                          {Object.entries(legendLayer.category_labels)
                            .sort(([left], [right]) => Number(left) - Number(right))
                            .map(([code, label]) => (
                              <span key={code}>
                                <i style={{ backgroundColor: legendLayer.category_colors?.[code] }} />
                                {legendLayer.product_id === "FAO_GAEZ_V5"
                                  ? label
                                  : crops.find((crop) => crop.id === label)?.name ?? label}
                              </span>
                            ))}
                        </div>
                      </>
                    )}
                    {!(legendLayer.category_labels && legendLayer.category_colors) && (
                      <>
                        <div className={`legend-ramp ${legendLayer.variable_kind === "quality" ? "quality-ramp" : ""}`} />
                        <div className="legend-range">
                          <span>
                            {legendLayer.variable_kind === "quality"
                              ? "Raw flag codes"
                              : legendLayer.render_range?.[0] ?? "Low"}
                          </span>
                          <span>
                            {legendLayer.variable_kind === "quality"
                              ? "Not interpreted"
                              : legendLayer.render_range?.[1] ?? "High"}
                          </span>
                        </div>
                      </>
                    )}
                    {legendLayer.quality_layers.length > 0 && !legendLayer.quality_filter_applied && (
                      <div className="qa-note">QA not applied · quality layers are separate</div>
                    )}
                  </div>
                ))}
              </>
            )}
            {!selectedTaskName && <p className="map-card-empty">Select a registered task to see its processed layers. Unavailable datasets are never filled with sample values.</p>}
          </aside>

          {inspection && (
            <aside className="map-card inspection-card">
              <button
                className="close-inspection"
                type="button"
                aria-label="Close pixel information"
                onClick={() => setInspection(null)}
              >
                ×
              </button>
              <div className="eyebrow">CELL INSPECTION</div>
              <strong>{String(inspection.variable ?? "Map location")}</strong>
              <div className="inspection-value">
                {inspection.scores && typeof inspection.scores === "object"
                  ? `Best crop: ${String(inspection.best_crop ?? "No data")}`
                  : inspection.value == null
                    ? String(inspection.message ?? "No data at this location")
                    : `${String(inspection.value)} ${inspection.units == null ? "" : String(inspection.units)}`}
              </div>
              {inspection.confidence != null && (
                <div className="inspection-date">
                  Confidence · {Math.round(Number(inspection.confidence) * 100)}%
                </div>
              )}
              {inspection.scores != null && typeof inspection.scores === "object" && (
                <div className="inspection-date">
                  {formatCropScores(inspection.scores, crops)}
                </div>
              )}
              {inspection.limiting_factors != null && (
                <div className="inspection-date">
                  Limiting factors · {formatLimitingFactors(inspection.limiting_factors, crops) || "None reported"}
                </div>
              )}
              {Array.isArray(inspection.missing_parameters) && inspection.missing_parameters.length > 0 && (
                <div className="inspection-date">
                  Missing variables · {inspection.missing_parameters.map(String).join(", ")}
                </div>
              )}
                {typeof inspection.observation_date === "string" && (
                <div className="inspection-date">
                  Observation · {String(inspection.observation_date)}
                </div>
              )}
              {(typeof inspection.lat === "number" || typeof inspection.lat === "string") && (
                <div className="inspection-date">
                  {Number(inspection.lat).toFixed(5)}° N, {Number(inspection.lon).toFixed(5)}° E
                </div>
              )}
            </aside>
          )}

          <div className="bottom-panel" id="tasks-panel" hidden={activeTab !== "Tasks"}>
            <div className="bottom-panel-header">
              <div>
                <span className="eyebrow">WORKFLOW</span>
                <h2>Collection tasks</h2>
              </div>
              <div className="task-actions">
                <button type="button" className="quiet-button" onClick={refreshLocalTasks}>
                  Refresh local
                </button>
                <button type="button" className="quiet-button" disabled={busy === "sync"} onClick={syncRemoteTasks}>
                  {busy === "sync" ? "Syncing…" : "Sync NASA"}
                </button>
              </div>
            </div>
            {tasks.length ? (
              <div className="task-table-wrap">
                <table className="task-table">
                  <thead>
                    <tr>
                      <th>AOI task</th>
                      <th>AOI · period</th>
                      <th>Task status</th>
                      <th>Download</th>
                      <th>Files</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {tasks.map((task) => (
                      <tr key={task.task_name ?? task.task_id ?? task.product_id}>
                        <td>
                          <strong>{task.task_name ?? task.product_id ?? "AppEEARS task"}</strong>
                          <small>{task.task_id ?? "NASA ID pending"}</small>
                        </td>
                        <td>
                          <span>{task.aoi_name ?? "Remote task"}</span>
                          <small>{task.start_date ?? "—"} – {task.end_date ?? "—"}</small>
                          {task.acquisition && (
                            <>
                              <small>
                                {task.acquisition.stage?.replaceAll("_", " ") ?? "Acquisition"} · {task.acquisition.status}
                              </small>
                              {Object.entries(task.acquisition.providers).map(([providerName, provider]) => (
                                <small key={providerName}>
                                  {providerName.replaceAll("_", " ")} · {provider.status}
                                  {provider.operations?.length ? ` · ${provider.operations.length} operation(s)` : ""}
                                  {provider.error ? ` · ${provider.error}` : ""}
                                </small>
                              ))}
                            </>
                          )}
                        </td>
                        <td><span className={`task-status status-${task.task_status}`}>{task.task_status}</span></td>
                        <td><span className="download-status">{task.download_status}</span></td>
                        <td>{task.downloaded_file_count}</td>
                        <td className="task-buttons">
                          {task.task_name && !task.remote_only && (
                            <button
                              type="button"
                              onClick={() => selectTask(task.task_name!)}
                              disabled={selectedTaskName === task.task_name}
                            >
                              {selectedTaskName === task.task_name ? "Selected" : "Select"}
                            </button>
                          )}
                          {task.task_type === "aoi_analysis" && (
                            <button
                              type="button"
                              onClick={() => downloadEverything(task)}
                              disabled={Boolean(busy)}
                              title="Check linked AppEEARS operations and prepare available data inside this AOI task"
                            >
                              {busy === task.task_id ? "Preparing data…" : "Prepare provider data"}
                            </button>
                          )}
                          {task.task_type === "aoi_analysis" ? (
                            <button
                              type="button"
                              onClick={refreshLocalTasks}
                              disabled={Boolean(busy)}
                            >
                              Refresh
                            </button>
                          ) : (
                            <button
                              type="button"
                              onClick={() => monitorTask(task.task_id)}
                              disabled={Boolean(busy) || !task.task_id}
                              title={task.remote_only ? "Query current remote status (not persisted locally)" : "Refresh task status"}
                            >
                              Check
                            </button>
                          )}
                          {task.task_type === "aoi_analysis" &&
                            Object.values(task.acquisition?.providers ?? {}).some(
                              (provider) => ["partial", "failed"].includes(provider.status),
                            ) && (
                              <button
                                type="button"
                                onClick={() => retryAoiTask(task)}
                                disabled={Boolean(busy)}
                                title="Retry only failed provider operations"
                              >
                                Retry failed providers
                              </button>
                            )}
                          {!task.remote_only && (
                            <button
                              type="button"
                              onClick={() => showTaskDetails(task.task_id)}
                              disabled={Boolean(busy)}
                            >
                              Details
                            </button>
                          )}
                          {task.task_type !== "aoi_analysis" && (
                            <button
                              type="button"
                              onClick={() => downloadTask(task.task_id)}
                              disabled={Boolean(busy) || task.task_status !== "completed" || task.remote_only}
                            >
                              Download
                            </button>
                          )}
                          {task.task_type !== "aoi_analysis" && !task.remote_only && task.task_id && task.download_status === "completed" && (
                            <button
                              type="button"
                              onClick={() => standardizeTask(task.task_id!)}
                              disabled={Boolean(busy) || task.preprocessing_status === "completed"}
                            >
                              {task.preprocessing_status === "completed" ? "Standardized" : "Standardize"}
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="no-tasks">
                <span className="no-task-symbol">↘</span>
                <div><strong>No collection tasks yet</strong><span>Requests submitted from this map will show here.</span></div>
              </div>
            )}
          </div>
          {taskDetails && (
            <aside className="map-card task-details-card">
              <button
                className="close-inspection"
                type="button"
                aria-label="Close task details"
                onClick={() => setTaskDetails(null)}
              >
                ×
              </button>
              <div className="eyebrow">TASK DETAILS</div>
              <strong>
                {taskDetails.task_type === "aoi_analysis"
                  ? String(taskDetails.task_name ?? "Unified AOI task")
                  : String(taskDetails.product_id ?? "AppEEARS task")}
              </strong>
              <div className="inspection-date">
                {String(taskDetails.start_date ?? "—")} – {String(taskDetails.end_date ?? "—")}
              </div>
              {Array.isArray(taskDetails.layers) && (
                <div className="inspection-date">
                  Layers · {taskDetails.layers.map(String).join(", ")}
                </div>
              )}
              {typeof taskDetails.error === "string" && taskDetails.error && (
                <div className="task-error">{taskDetails.error}</div>
              )}
              {taskAcquisitionDetails && (
                <div className="task-provider-details">
                  <strong>
                    Acquisition · {String(taskAcquisitionDetails.status ?? "unknown")}
                    {typeof taskAcquisitionDetails.stage === "string"
                      ? ` · ${taskAcquisitionDetails.stage.replaceAll("_", " ")}`
                      : ""}
                  </strong>
                  {taskProviderDetails && Object.entries(taskProviderDetails).map(([name, value]) => {
                    const provider = objectRecord(value);
                    if (!provider) return null;
                    const operations = Array.isArray(provider.operations)
                      ? provider.operations
                      : [];
                    return (
                      <div key={name}>
                        <span>{name.replaceAll("_", " ")} · {String(provider.status ?? "unknown")}</span>
                        {Array.isArray(provider.variables) && provider.variables.length > 0 && (
                          <small>Datasets · {provider.variables.map(String).join(", ")}</small>
                        )}
                        {typeof provider.error === "string" && provider.error && (
                          <span className="task-error">{provider.error}</span>
                        )}
                        {Array.isArray(provider.failures) && provider.failures.map((failureValue, index) => {
                          const failure = objectRecord(failureValue);
                          if (!failure) return null;
                          return (
                            <span className="task-error" key={`${String(failure.product_id)}-${index}`}>
                              {String(failure.product_id ?? "Provider operation")} · {String(failure.error ?? "failed")}
                            </span>
                          );
                        })}
                        {operations.map((operationValue) => {
                          const operation = objectRecord(operationValue);
                          if (!operation) return null;
                          return (
                            <small key={String(operation.task_id ?? operation.task_name)}>
                              {String(operation.product_id ?? "Provider operation")} · {String(operation.status ?? "unknown")} · {String(operation.task_id ?? "")}
                            </small>
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              )}
            </aside>
          )}
        </section>
      </div>

      <aside className="future-panel" hidden={activeTab !== "Crop suitability"}>
        <div className="future-panel-content">
          <span className="future-icon">⌁</span>
          <div className="eyebrow">TASK-SCOPED ANALYSIS</div>
          <h2>Crop suitability</h2>
          <p>
            GAEZ v5 provides the primary long-term suitability screening for the selected AOI.
            NASA observations remain separate context and are not combined into its index.
          </p>
          <section className="gaez-panel" aria-label="GAEZ v5 historical crop suitability">
            <div className="eyebrow">FAO / IIASA · HISTORICAL BASELINE</div>
            <h3>GAEZ v5 baseline index ranking</h3>
            <p>
              The SXX index is a long-term modeled agro-ecological indicator, not a probability
              of success, yield forecast, or guarantee. It is not combined with NASA observations
              or legacy environmental scoring. Rain-fed and irrigated results are ranked separately.
            </p>
            <label className="task-name-field">
              <span>WATER REGIME</span>
              <select
                value={gaezWaterRegime}
                onChange={(event) => setGaezWaterRegime(
                  event.target.value as "rainfed" | "irrigated" | "both",
                )}
              >
                <option value="both">Both (separate rankings)</option>
                <option value="rainfed">Rain-fed only</option>
                <option value="irrigated">Irrigated only</option>
              </select>
            </label>
            <label className="task-name-field">
              <span>INPUT / MANAGEMENT LEVEL</span>
              <select
                value={gaezManagementLevel}
                onChange={(event) => setGaezManagementLevel(
                  event.target.value as "low" | "high",
                )}
              >
                <option value="high">High input</option>
                <option value="low">Low input</option>
              </select>
            </label>
            <label className="gaez-crop-search">
              <span className="sr-only">Search the crop catalog</span>
              <input
                type="search"
                value={gaezCropSearch}
                onChange={(event) => setGaezCropSearch(event.target.value)}
                placeholder="Search crops or categories"
              />
            </label>
            <div className="gaez-catalog-source">
              {gaezCatalog ? (
                <>
                  Validated products: {gaezCatalog.provider} · {gaezCatalog.period} ·{" "}
                  {gaezCatalog.inventory_snapshot.scenario_code} ·{" "}
                  {gaezCatalog.inventory_snapshot.climate_code} · inventory checked{" "}
                  {gaezCatalog.inventory_snapshot.verified_on} ·{" "}
                  <a href={gaezCatalog.source} target="_blank" rel="noreferrer">source</a>.
                  AOI coverage is measured after analysis; missing data are not ranked as unsuitable.
                </>
              ) : "Loading GAEZ crop catalog…"}
            </div>
            <div className="gaez-crop-catalog" aria-label="GAEZ crop catalog">
              {gaezCatalog ? gaezCropGroups.map(([category, categoryCrops]) => (
                <section key={category} className="gaez-crop-category">
                  <h4>{category}</h4>
                  {categoryCrops.map((crop) => {
                    if ("code" in crop) {
                      const managementCode = gaezManagementLevel === "high" ? "H" : "L";
                      const regimesToShow = [
                        ["Rain-fed", `${managementCode}RLM`],
                        ["Irrigated", `${managementCode}ILM`],
                      ].filter(([name]) =>
                        gaezWaterRegime === "both" ||
                        (gaezWaterRegime === "rainfed" && name === "Rain-fed") ||
                        (gaezWaterRegime === "irrigated" && name === "Irrigated"),
                      );
                      const availableRegimes = regimesToShow
                        .filter(([, code]) =>
                          crop.available_scenario_codes.includes(code),
                        )
                        .map(([name]) => name);
                      return (
                        <label key={crop.code} className="gaez-crop-option">
                          <span className="gaez-crop-option-main">
                            <input
                              type="checkbox"
                              checked={gaezCropCodes.has(crop.code)}
                              onChange={() => setGaezCropCodes((current) => {
                                const next = new Set(current);
                                if (next.has(crop.code)) next.delete(crop.code);
                                else next.add(crop.code);
                                return next;
                              })}
                            />
                            <span>{crop.name}</span>
                          </span>
                          <span className={`gaez-availability ${crop.availability_status}`}>
                            {crop.availability_status === "available" ? "Available" : "Partial"}
                          </span>
                          <small>
                            {availableRegimes.length
                              ? `${availableRegimes.join(" + ")} · ${gaezManagementLevel}-input data; AOI coverage pending${availableRegimes.length < regimesToShow.length ? ` · no product for ${regimesToShow.filter(([name]) => !availableRegimes.includes(name)).map(([name]) => name).join(", ")}` : ""}`
                              : `No ${gaezManagementLevel}-input product for the selected water regime; no score will be assigned`}
                          </small>
                          <small>Source: GAEZ v5 SXX/SIX · {gaezCatalog?.period}</small>
                        </label>
                      );
                    }
                    return (
                      <div key={crop.id} className="gaez-crop-option unavailable">
                        <span className="gaez-crop-option-main">{crop.name}</span>
                        <span className="gaez-availability unavailable">Unavailable</span>
                        <small>{crop.availability_reason}</small>
                        <small>Source: no validated GAEZ v5 suitability product in the checked inventory</small>
                      </div>
                    );
                  })}
                </section>
              )) : <p className="future-footnote">Loading crop categories…</p>}
              {gaezCatalog && !gaezCropGroups.length && (
                <p className="future-footnote">No crops match this search.</p>
              )}
            </div>
            <button
              type="button"
              className="primary-button"
              onClick={runGaezAnalysis}
              disabled={
                !selectedTask?.task_id ||
                !gaezCatalog ||
                !gaezCropCodes.size ||
                Boolean(busy)
              }
            >
              {busy === "gaez" ? "Acquiring GAEZ windows…" : "Rank crops for this AOI"}
              <span>→</span>
            </button>
            {gaezResult && (
              <div className="gaez-results" aria-live="polite">
                <strong>
                  GAEZ status: {gaezResult.status.replaceAll("_", " ")}
                </strong>
                {gaezResult.analysis_outdated && (
                  <p className="future-footnote">
                    This saved result predates the current GAEZ processing and coverage checks.
                    Run the analysis again to refresh it; old results are retained for traceability.
                  </p>
                )}
                {["queued", "running"].includes(gaezResult.status) && (
                  <ul className="gaez-operation-list">
                    {(gaezResult.operations ?? []).map((operation, index) => (
                      <li key={`${operation.water_regime}-${operation.crop_code}-${index}`}>
                        {operation.crop_name} · {operation.water_regime} · {operation.status}
                        {operation.error ? ` · ${operation.error}` : ""}
                      </li>
                    ))}
                    {!gaezResult.operations?.length && <li>Preparing source-window requests…</li>}
                  </ul>
                )}
                {gaezResult.status === "failed" && (
                  <p className="future-footnote">{gaezResult.error ?? "GAEZ acquisition failed."}</p>
                )}
                {gaezResult.scenarios && gaezResult.scenarios.length > 0 && (
                  <>
                    {gaezResult.scenarios.length > 1 && (
                      <label className="task-name-field">
                        <span>RESULT SCENARIO</span>
                        <select
                          value={gaezResultRegime}
                          onChange={(event) => setGaezResultRegime(
                            event.target.value as "rainfed" | "irrigated",
                          )}
                        >
                          {gaezResult.scenarios.map((scenario) => (
                            <option key={scenario.water_regime} value={scenario.water_regime}>
                              {scenario.water_regime === "rainfed" ? "Rain-fed" : "Irrigated"}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    <p className="future-footnote">
                      {activeGaezScenario?.water_regime === "rainfed" ? "Rain-fed" : "Irrigated"} ·{" "}
                      {activeGaezScenario?.management_level} input
                      {activeGaezScenario?.management_code
                        ? ` (${activeGaezScenario.management_code})`
                        : ""} · {gaezResult.period} ·{" "}
                      {gaezResult.scenario_code ?? "scenario metadata unavailable"} ·{" "}
                      {gaezResult.climate_code ?? "climate metadata unavailable"} ·{" "}
                      {gaezResult.period_code ?? "period metadata unavailable"} · SXX{" "}
                      {gaezResult.continuous_product?.url ? (
                        <a href={gaezResult.continuous_product.url} target="_blank" rel="noreferrer">
                          {activeGaezScenario?.product_identifiers?.continuous_index
                            ?? gaezResult.continuous_product.mapset}
                        </a>
                      ) : activeGaezScenario?.product_identifiers?.continuous_index
                        ?? "RES05-SXX30AS"} ·{" "}
                      SIX{" "}
                      {gaezResult.class_product?.url ? (
                        <a href={gaezResult.class_product.url} target="_blank" rel="noreferrer">
                          {activeGaezScenario?.product_identifiers?.official_classes
                            ?? gaezResult.class_product.mapset}
                        </a>
                      ) : activeGaezScenario?.product_identifiers?.official_classes
                        ?? "RES05-SIX30AS"} ·{" "}
                      EPSG:4326 · approx. {(gaezResult.spatial_resolution_degrees ?? 0) * 111.32} km at equator
                    </p>
                    <p className="future-footnote">
                      SXX units: {gaezResult.index_metadata?.units ?? "GAEZ suitability index (0-10000)"} ·{" "}
                      raw range {gaezResult.index_metadata?.valid_range?.join("–") ?? "0–10000"} ·{" "}
                      scale {gaezResult.index_metadata?.scale_factor ?? 1} ·{" "}
                      no-data {gaezResult.index_metadata?.no_data ?? -9}. Coverage is data availability,
                      not agronomic suitability.
                    </p>
                    {gaezRankedCrops.length > 0 ? (
                      <>
                        <h4>Highest-supported crop: {gaezRankedCrops[0].crop_name}</h4>
                        <ol className="gaez-ranking-list">
                          {gaezRankedCrops.slice(0, 5).map((crop) => (
                            <li key={crop.crop_code}>
                              <div className="gaez-ranking-row">
                                <span>
                                  {crop.rank}. {crop.crop_name} ({crop.crop_code})
                                  <small>
                                    SXX area-weighted mean {crop.mean_suitability_index?.toFixed(1) ?? "No data"} / 10,000
                                    {" · "}valid non-water data {(crop.valid_data_coverage * 100).toFixed(1)}%
                                    {crop.valid_data_coverage_of_aoi != null
                                      ? ` of AOI ${(crop.valid_data_coverage_of_aoi * 100).toFixed(1)}%`
                                      : ""}
                                    {crop.intersecting_cell_count != null
                                      ? ` · ${crop.valid_index_cell_count ?? "?"}/${crop.intersecting_cell_count} valid SXX cells`
                                      : ""}
                                    {crop.water_area_share_of_aoi != null
                                      ? ` · water ${(crop.water_area_share_of_aoi * 100).toFixed(1)}% of AOI`
                                      : ""}
                                    {crop.class_data_available && crop.class_distribution.length
                                      ? (() => {
                                          const dominantClass = [...crop.class_distribution]
                                            .sort((first, second) => second.area_km2 - first.area_km2)[0];
                                          return dominantClass.area_km2 > 0
                                            ? ` · dominant official class ${dominantClass.label} (${((dominantClass.share_of_aoi ?? 0) * 100).toFixed(1)}% of AOI)`
                                            : "";
                                        })()
                                      : ""}
                                  </small>
                                </span>
                                <button
                                  type="button"
                                  className="quiet-button"
                                  onClick={() => {
                                    showGaezMap(crop);
                                  }}
                                >
                                  Show suitability map
                                </button>
                              </div>
                              <details>
                                <summary>Official SIX class distribution</summary>
                                {(crop.class_data_available
                                  ?? crop.class_distribution.some(
                                    (item) => item.share_of_classified_area != null,
                                  )) ? (
                                  <>
                                    <p className="future-footnote">
                                      Area and share of the AOI per official class.
                                      {gaezResult.class_metadata
                                        ? ` SIX scale ${gaezResult.class_metadata.scale_factor}, no-data ${gaezResult.class_metadata.no_data}.`
                                        : ""}
                                      {crop.valid_class_cell_count != null
                                        ? ` Class data: ${crop.valid_class_cell_count}/${crop.intersecting_cell_count ?? "?"} intersecting cells`
                                        : ""}
                                      {crop.class_data_coverage != null
                                        ? ` (${(crop.class_data_coverage * 100).toFixed(1)}% AOI coverage).`
                                        : "."}
                                    </p>
                                    <ul>
                                      {crop.class_distribution.map((item) => (
                                      <li key={item.class_code}>
                                        {item.label}: {item.area_km2.toFixed(2)} km²
                                        {" · "}{item.share_of_aoi != null
                                          ? `${(item.share_of_aoi * 100).toFixed(1)}% of AOI`
                                          : item.share_of_classified_area == null
                                            ? "no classified area"
                                            : `${(item.share_of_classified_area * 100).toFixed(1)}% of classified area`}
                                      </li>
                                      ))}
                                    </ul>
                                  </>
                                ) : (
                                  <p className="future-footnote">
                                    Official SIX class data is unavailable for this crop and scenario.
                                    {crop.class_data_error ? ` ${crop.class_data_error}` : ""}
                                  </p>
                                )}
                              </details>
                            </li>
                          ))}
                        </ol>
                      </>
                    ) : (
                      <p className="future-footnote">
                        No crop was rankable for this scenario. Products unavailable for the selected scenario and crops below the configured {(gaezResult.minimum_coverage_for_ranking ?? 0) * 100}% AOI data-coverage threshold are reported separately; missing data are not interpreted as unsuitable.
                      </p>
                    )}
                    {activeGaezScenario?.unranked_crops.map((crop) => (
                      <p className="future-footnote" key={crop.crop_code}>
                        {crop.crop_name} is not ranked: {crop.ranking_exclusion_reason}
                      </p>
                    ))}
                    {gaezResult.failures?.length ? (
                      <ul className="gaez-operation-list" aria-label="GAEZ data limitations">
                        {gaezResult.failures.map((item, index) => (
                          <li key={`${item.crop_code}-${item.water_regime}-${item.product ?? "SXX"}-${index}`}>
                            {item.crop_name ?? item.crop_code} ({item.crop_code}) · {item.water_regime} · {item.management_level} ·{" "}
                            {item.product ?? "operation"} {item.failure_type ?? "unavailable"}
                            {item.failure_stage ? ` at ${item.failure_stage}` : ""}
                            {`: ${item.error}`}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                    <p className="future-footnote">
                      Source: <a href={gaezResult.legend_url} target="_blank" rel="noreferrer">official GAEZ v5 legend</a>.
                      {" "}{gaezResult.attribution}
                    </p>
                    {gaezResult.limitations?.map((limitation) => (
                      <small className="gaez-limitation" key={limitation}>{limitation}</small>
                    ))}
                  </>
                )}
              </div>
            )}
          </section>
          <div className="suitability-input-note">
            <strong>Legacy environmental scoring is separate from GAEZ.</strong>
            <span>{suitabilityInputs?.status ?? "Checking the crop scoring criteria against spatial raster inputs…"}</span>
            {suitabilityInputs?.crop_reference_scoring_criteria.length ? (
              <small>
                Current crop reference criteria: {suitabilityInputs.crop_reference_scoring_criteria.join(", ")}.
              </small>
            ) : null}
          </div>
          <div className="suitability-modes" role="group" aria-label="Suitability result display mode">
            <button
              type="button"
              aria-pressed={suitabilityMode === "best-crop"}
              disabled={!bestCropRaster}
              onClick={() => showSuitabilityMode("best-crop")}
            >
              Best Crop
            </button>
            <button
              type="button"
              aria-pressed={suitabilityMode === "individual-score"}
              disabled={!scoreRasters.length}
              onClick={() => showSuitabilityMode("individual-score")}
            >
              Individual Crop Score
            </button>
          </div>
          {suitabilityMode === "individual-score" && scoreRasters.length > 0 && (
            <label className="task-name-field">
              <span>CROP SCORE LAYER</span>
              <select
                value={scoreRasters.some((layer) => layer.variable === selectedScoreCropId)
                  ? selectedScoreCropId
                  : scoreRasters[0].variable}
                onChange={(event) => setSelectedScoreCropId(event.target.value)}
              >
                {scoreRasters.map((layer) => (
                  <option key={layer.id} value={layer.variable}>
                    {crops.find((crop) => crop.id === layer.variable)?.name ?? layer.name}
                  </option>
                ))}
              </select>
              <button
                className="quiet-button"
                type="button"
                onClick={() => showSuitabilityMode("individual-score")}
              >
                Show selected crop score and confidence
              </button>
            </label>
          )}
          {suitabilityRasterRecords.length === 0 && (
            <p className="future-footnote">
              Best Crop, individual scores, confidence, and cell diagnostics appear here after a valid task-scoped analysis exists.
            </p>
          )}
          <label className="task-name-field">
            <span>NASA DATA TASK</span>
            <select
              value={selectedTaskName}
              onChange={(event) => selectTask(event.target.value)}
            >
              <option value="">Select a registered task</option>
              {tasks.filter((task) => task.task_name && !task.remote_only).map((task) => (
                <option key={task.task_name} value={task.task_name!}>
                  {task.task_name} · {task.task_status}
                </option>
              ))}
            </select>
          </label>
          {selectedTask && (
            <div className="future-footnote">
              NASA task ID: {selectedTask.task_id ?? "not assigned"}
              <br />
              Download: {selectedTask.download_status} · Preprocessing: {selectedTask.preprocessing_status ?? "pending"}
            </div>
          )}
          <label className="task-name-field">
            <span>ANALYSIS RUN NAME · OPTIONAL</span>
            <input
              value={analysisName}
              onChange={(event) => setAnalysisName(event.target.value)}
              placeholder="Generated as ANALYSIS_001, etc."
              maxLength={120}
            />
          </label>
          <div className="crop-chips crop-selection-list">
            {crops.map((crop) => (
              <label key={crop.id}>
                <input
                  type="checkbox"
                  checked={selectedCropIds.has(crop.id)}
                  onChange={() => setSelectedCropIds((current) => {
                    const next = new Set(current);
                    if (next.has(crop.id)) next.delete(crop.id);
                    else next.add(crop.id);
                    return next;
                  })}
                />
                {crop.name}
              </label>
            ))}
          </div>
          <button
            type="button"
            className="primary-button"
            onClick={runSuitabilityAnalysis}
            disabled={
              !selectedTaskName ||
              selectedTask?.preprocessing_status !== "completed" ||
              !suitabilityAvailable ||
              !suitabilityInputs ||
              suitabilityInputs.matching_raster_scoring_criteria.length === 0 ||
              Boolean(busy)
            }
          >
            {busy === "suitability" ? "Analyzing…" : "Run new analysis"}
            <span>→</span>
          </button>
          {rasters.some((layer) => layer.product_id === "crop-suitability") && (
            <div className="future-footnote">
              Map layers for {selectedTaskName} are available in the map layer control.
              Suitability scores use a 0–100 scale; confidence is a separate layer.
            </div>
          )}
          <div className="future-footnote">
            Crop requirement guidance is general and not locally validated. A result is screening information, not an agronomic recommendation.
          </div>
        </div>
      </aside>

      {notice && (
        <div className={`notice notice-${notice.kind}`} role="status">
          <span>{notice.kind === "error" ? "!" : notice.kind === "success" ? "✓" : "i"}</span>
          <p>{notice.message}</p>
          <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss notification">×</button>
        </div>
      )}
    </main>
  );
}

export default App;
