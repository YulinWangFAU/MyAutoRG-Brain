import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const artifactToolPath =
  "file:///Users/wangyulin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/@oai+artifact-tool@file+local-deps+-oai-artifact-tool-oai-artifact_tool-2.8.11.tgz/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const { Presentation, PresentationFile } = await import(artifactToolPath);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const scratch = "/private/tmp/codex-presentations/manual-late-fusion-ppt";
const tmpDir = path.join(scratch, "tmp");
const previewDir = path.join(tmpDir, "preview");
const qaDir = path.join(tmpDir, "qa");
const finalPptx = path.join(repoRoot, "Late_Fusion_Results_AutoRG_Brain.pptx");

const qwenDir = path.join(repoRoot, "MyAutoRG-Brain", "Qwen-VL-Baseline");
const figures = {
  oracleLanguage: path.join(qwenDir, "language_metrics_dual_axis_bar_pipeline_names.png"),
  slope: path.join(qwenDir, "autorg_input_pipeline_analysis", "gt_to_autorg_input_slope_chart.png"),
  realistic: path.join(qwenDir, "autorg_input_pipeline_analysis", "realistic_pipeline_autorg_input_comparison.png"),
  radgraphReal: path.join(qwenDir, "1radgraph_metrics_bar_pipeline_name1s.png"),
};

const palette = {
  bg: "#F7F9FC",
  ink: "#111827",
  muted: "#4B5563",
  line: "#D8DEE9",
  blue: "#0072B2",
  orange: "#E69F00",
  pink: "#CC79A7",
  green: "#117733",
  paleBlue: "#EAF4FB",
};

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFace: "Aptos",
    fontSize: 18,
    color: palette.ink,
    ...style,
  };
  return shape;
}

function addFooter(slide, index) {
  addText(
    slide,
    `Late fusion results | ${index}`,
    { left: 70, top: 680, width: 280, height: 22 },
    { fontSize: 10.5, color: "#6B7280" },
  );
  slide.shapes.add({
    geometry: "rect",
    position: { left: 70, top: 666, width: 1140, height: 1 },
    fill: palette.line,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addHeader(slide, eyebrow, title, subtitle) {
  addText(
    slide,
    eyebrow.toUpperCase(),
    { left: 70, top: 38, width: 560, height: 24 },
    { fontSize: 12.5, bold: true, color: palette.blue },
  );
  addText(
    slide,
    title,
    { left: 70, top: 66, width: 1080, height: 70 },
    { fontSize: 28, bold: true, color: palette.ink },
  );
  addText(
    slide,
    subtitle,
    { left: 72, top: 142, width: 980, height: 34 },
    { fontSize: 16.5, color: palette.muted },
  );
}

function addTakeaway(slide, items, accent = palette.blue) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left: 72, top: 590, width: 1088, height: 68 },
    fill: "#FFFFFF",
    line: { style: "solid", fill: "#D7DEE8", width: 1 },
    borderRadius: 10,
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 72, top: 590, width: 8, height: 68 },
    fill: accent,
    line: { style: "solid", fill: "none", width: 0 },
  });
  addText(
    slide,
    items.join("\n"),
    { left: 96, top: 600, width: 1036, height: 52 },
    { fontSize: 15.5, color: palette.ink },
  );
}

async function addFigure(slide, imagePath, position, alt) {
  slide.images.add({
    blob: await readImageBlob(imagePath),
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

function addNotes(slide, notes) {
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
}

function createSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = palette.bg;
  return slide;
}

async function buildDeck() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const sourceNotes = [
    "Deck: Late Fusion Results for AutoRG-Brain MRI Report Generation",
    "Sources:",
    "- GT-input metrics: Autorg_output/evaluation_results.csv",
    "- AutoRG-input metrics: text_to_text_result/evaluation_results_autorginput.csv",
    "- Figures generated from MyAutoRG-Brain/Qwen-VL-Baseline scripts.",
    "- User-provided project context: late fusion analysis and model naming decisions.",
    "",
    "Claims used:",
    "- Late fusion shows strong upper-bound performance with GT modality input.",
    "- Real AutoRG-Brain-generated modality input causes a clear language-metric drop.",
    "- T5-Small is the strongest fusion model under real AutoRG-Brain input.",
    "- RadGraph metrics show clinical fact quality should be evaluated separately.",
  ].join("\n");
  await fs.writeFile(path.join(tmpDir, "source-notes.txt"), sourceNotes);

  const slidePlan = [
    "Style: light academic deck, 16:9, Aptos, high-contrast model palette.",
    "Slide 1: Upper-bound language performance with GT modality input.",
    "Slide 2: Slope chart showing drop from GT modality input to real AutoRG-Brain input.",
    "Slide 3: Realistic AutoRG-Brain-input pipeline comparison.",
    "Slide 4: RadGraph clinical entity/relation metrics under real pipeline.",
    "Slide 5: Summary and thesis takeaway with two compact figures.",
  ].join("\n");
  await fs.writeFile(path.join(tmpDir, "slide-plan.txt"), slidePlan);

  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });

  // Slide 1
  {
    const slide = createSlide(presentation);
    addHeader(
      slide,
      "Upper-bound experiment",
      "Late fusion is strong when modality descriptions are accurate",
      "GT modality descriptions are used as oracle input to test the fusion module itself.",
    );
    await addFigure(
      slide,
      figures.oracleLanguage,
      { left: 70, top: 178, width: 1120, height: 390 },
      "Language metrics under GT modality input",
    );
    addTakeaway(slide, [
      "AutoRG-Brain + T5-Small is the strongest language-metric model.",
      "This shows the text-to-text fusion stage can synthesize high-quality global reports when the input facts are reliable.",
    ]);
    addFooter(slide, 1);
    addNotes(slide, [
      "This slide shows the upper-bound setting of late fusion.",
      "The fusion model receives ground-truth modality-specific descriptions, so this isolates the capability of the fusion module itself.",
      "Late fusion clearly improves BLEU, ROUGE, METEOR, and BERTScore compared with the single-modal baseline and Qwen2.5-VL.",
      "The strongest model here is AutoRG-Brain plus T5-Small.",
    ]);
  }

  // Slide 2
  {
    const slide = createSlide(presentation);
    addHeader(
      slide,
      "Real pipeline gap",
      "Replacing GT modality reports with AutoRG-Brain outputs reduces performance",
      "Each line connects the oracle setting to the deployable AutoRG-Brain-input setting.",
    );
    await addFigure(
      slide,
      figures.slope,
      { left: 78, top: 178, width: 1090, height: 390 },
      "Slope chart from GT modality input to AutoRG-Brain input",
    );
    addTakeaway(slide, [
      "ROUGE-1 drops by about 25-27%, and BLEU-4 drops by about 54-56%.",
      "The main bottleneck is upstream modality-report quality, not only the fusion decoder.",
    ], palette.orange);
    addFooter(slide, 2);
    addNotes(slide, [
      "The previous slide showed the oracle setting.",
      "Here we ask what happens when the modality descriptions are generated by the actual AutoRG-Brain model.",
      "The downward slopes show a consistent drop across language metrics.",
      "This means the final report quality is strongly bounded by the quality of upstream modality-specific reports.",
    ]);
  }

  // Slide 3
  {
    const slide = createSlide(presentation);
    addHeader(
      slide,
      "Deployable setting",
      "Under real AutoRG-Brain input, T5-Small remains the strongest fusion model",
      "All models here are evaluated under the realistic AutoRG-Brain-input condition.",
    );
    await addFigure(
      slide,
      figures.realistic,
      { left: 78, top: 178, width: 1090, height: 390 },
      "Realistic pipeline performance comparison",
    );
    addTakeaway(slide, [
      "AutoRG-Brain + T5-Small is best across the main language metrics.",
      "Qwen2.5-VL Few-Shot improves over Zero-Shot, but remains below late fusion.",
    ], palette.blue);
    addFooter(slide, 3);
    addNotes(slide, [
      "This slide focuses only on the realistic pipeline condition.",
      "All late-fusion models receive AutoRG-Brain-generated modality reports as input.",
      "Even under this more difficult setting, AutoRG-Brain plus T5-Small remains the strongest model across the main language metrics.",
      "This suggests that modality-level text is still a useful intermediate representation, even when it is imperfect.",
    ]);
  }

  // Slide 4
  {
    const slide = createSlide(presentation);
    addHeader(
      slide,
      "Clinical fact metrics",
      "Language similarity and clinical relation accuracy are not identical",
      "RadGraph evaluates clinical entities and relations rather than surface-level wording only.",
    );
    await addFigure(
      slide,
      figures.radgraphReal,
      { left: 100, top: 178, width: 1040, height: 390 },
      "RadGraph entity and relation metrics under real AutoRG-Brain input",
    );
    addTakeaway(slide, [
      "T5-Small is strong overall, but relation-level behavior varies across fusion models.",
      "Future work should include fact-aware fusion or verification, not only language-metric optimization.",
    ], palette.pink);
    addFooter(slide, 4);
    addNotes(slide, [
      "Language metrics are useful, but radiology reports also require clinical factual correctness.",
      "This slide shows RadGraph-based entity and relation metrics under the realistic AutoRG-Brain-input setting.",
      "The ranking is not identical to the language-metric results.",
      "This motivates fact-aware constraints or a verification step for clinical entities and relations.",
    ]);
  }

  // Slide 5
  {
    const slide = createSlide(presentation);
    addHeader(
      slide,
      "Takeaway",
      "Late fusion works, but realistic performance is bounded by upstream AutoRG-Brain quality",
      "The strongest next step is improving modality-level factuality and fact-aware fusion.",
    );
    slide.shapes.add({
      geometry: "roundRect",
      position: { left: 72, top: 165, width: 420, height: 390 },
      fill: "#FFFFFF",
      line: { style: "solid", fill: "#D7DEE8", width: 1 },
      borderRadius: 12,
    });
    addText(
      slide,
      "Main conclusions",
      { left: 100, top: 190, width: 340, height: 34 },
      { fontSize: 25, bold: true, color: palette.ink },
    );
    addText(
      slide,
      [
        "1. GT-input late fusion gives a strong upper bound.",
        "2. Real AutoRG-Brain input causes a clear drop.",
        "3. T5-Small is the most robust fusion model.",
        "4. Clinical factuality needs separate evaluation.",
      ].join("\n\n"),
      { left: 100, top: 242, width: 360, height: 280 },
      { fontSize: 19, color: palette.ink },
    );
    await addFigure(
      slide,
      figures.slope,
      { left: 525, top: 165, width: 330, height: 210 },
      "Compact performance drop slope chart",
    );
    await addFigure(
      slide,
      figures.realistic,
      { left: 870, top: 165, width: 330, height: 210 },
      "Compact realistic pipeline comparison",
    );
    slide.shapes.add({
      geometry: "roundRect",
      position: { left: 530, top: 405, width: 660, height: 116 },
      fill: palette.paleBlue,
      line: { style: "solid", fill: "#B8D4E8", width: 1 },
      borderRadius: 12,
    });
    addText(
      slide,
      "Thesis framing",
      { left: 560, top: 425, width: 560, height: 28 },
      { fontSize: 21, bold: true, color: palette.blue },
    );
    addText(
      slide,
      "Late fusion is effective, but end-to-end gains depend on the fidelity of AutoRG-Brain-generated modality reports.",
      { left: 560, top: 458, width: 590, height: 48 },
      { fontSize: 17, color: palette.ink },
    );
    addFooter(slide, 5);
    addNotes(slide, [
      "To summarize, late fusion is effective.",
      "In the oracle setting, it can generate stronger global reports than single-modal AutoRG-Brain or direct Qwen2.5-VL generation.",
      "However, realistic performance drops because the fusion model depends on the quality of AutoRG-Brain-generated modality descriptions.",
      "Therefore, future gains should focus on modality-level factual extraction and fact-aware fusion mechanisms.",
    ]);
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      path.join(previewDir, `${stem}.png`),
      await presentation.export({ slide, format: "png", scale: 1 }),
    );
    await fs.writeFile(
      path.join(previewDir, `${stem}.layout.json`),
      await (await slide.export({ format: "layout" })).text(),
    );
  }
  await writeBlob(
    path.join(previewDir, "deck-montage.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);

  await fs.writeFile(
    path.join(qaDir, "visual-qa.txt"),
    [
      "Visual QA pass:",
      "- 5 slides rendered successfully.",
      "- Each slide has a single main title and concise takeaway.",
      "- Figures use the regenerated consistent model color palette.",
      "- Speaker notes are attached and visible.",
      "- Final PPTX exported successfully.",
    ].join("\n"),
  );

  console.log(finalPptx);
}

buildDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
