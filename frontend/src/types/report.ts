export type AttributionStage = {
  name: string;
  label: string | null;
  confidence: number | null;
  metrics: Record<string, any>;
  visuals: Record<string, string>;
};

export type Report = {
  result: {
    label: string;
    score_ai: number | null;
    components: Record<string, number | string>;
  };
  metrics: Record<string, any>;
  visuals: Record<string, string>;
  exif: Record<string, any>;

  attribution?: {
    generator?: AttributionStage;
    sd_variant?: AttributionStage;
    [key: string]: AttributionStage | undefined;
  };
};

export type MethodsResp = {
  methods: {
    key: string;
    description: string;
    how_text?: string;
    how_title?: string;
  }[];
};
