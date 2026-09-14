-- theorem-numbering.lua
-- Pandoc Lua filter for automatic theorem/definition numbering across decker decks.
--
-- How it works:
--   1. On every invocation, scans ALL *-deck.md files in the same directory
--   2. Parses YAML frontmatter for `chapter` and `subtitle`, counts theorem-like divs per file
--   3. Sorts files by name, groups by chapter, computes cumulative offsets
--   4. Identifies the current file by matching `subtitle` from document metadata
--   5. Walks the current file's AST: prepends bold labels, sets div IDs, resolves references
--
-- Supported div classes: definition, theorem, lemma, corollary, algorithm, axiom
-- Supported extra classes: .inline (prepend to first <p>), .noinc (don't increment counter)
-- Labels: ::: {.definition #my-label}  →  id="my-label" in HTML
-- References: [](#my-label)  →  filled with "Definition 1.3" (or cross-file with full URL)

-- German display names (matching the existing JS labels)
local DISPLAY_NAMES = {
  definition = "Definition",
  theorem    = "Satz",
  lemma      = "Lemma",
  corollary  = "Korollar",
  algorithm  = "Algorithmus",
  axiom      = "Axiom",
}

local THEOREM_CLASSES = {}
for k, _ in pairs(DISPLAY_NAMES) do
  THEOREM_CLASSES[k] = true
end

-- Detect the theorem class of a Div element (returns class name or nil)
local function get_theorem_class(div)
  for _, cls in ipairs(div.classes) do
    if THEOREM_CLASSES[cls] then
      return cls
    end
  end
  return nil
end

--------------------------------------------------------------------------------
-- Phase 1: Scan all deck files to build the numbering scheme
--------------------------------------------------------------------------------

-- Find the directory containing the deck files.
-- Try PANDOC_STATE.input_files first; if empty (decker feeds via stdin),
-- fall back to PANDOC_STATE.resource_path or current directory.
local function get_deck_dir()
  local input = PANDOC_STATE.input_files[1]
  if input then
    local dir = input:match("(.*/)")
    return dir or "./"
  end
  -- Decker sets resource_path to "." — use current working directory
  return "./"
end

-- Parse YAML frontmatter from raw text; returns chapter (number) and subtitle (string)
local function parse_frontmatter(text)
  local front = text:match("^%-%-%-\n(.-)%-%-%-")
  if not front then return nil, nil end
  local ch = front:match("chapter:%s*(%d+)")
  local sub = front:match("subtitle:%s*(.-)%s*\n")
  return ch and tonumber(ch) or nil, sub
end

-- Count theorem-like div openings in raw markdown text.
-- Returns: total count (excluding .noinc), list of {label, class, noinc} entries,
--          and slide_groups (list of lists: labeled IDs grouped by slide).
local function count_theorems_raw(text)
  local count = 0
  local entries = {}
  local slide_groups = {}       -- { {"label-a", "label-b"}, {"label-c"}, ... }
  local current_slide_labels = {}

  for line in text:gmatch("[^\n]+") do
    -- Detect slide boundary (level-1 header: single # not followed by another #)
    if line:match("^#[^#]") or line == "#" then
      if #current_slide_labels > 0 then
        table.insert(slide_groups, current_slide_labels)
      end
      current_slide_labels = {}
    end

    -- Match both `::: definition` and `::: {.definition #label .inline .noinc}`
    local attrs = line:match("^:::%s*{(.-)}")
    if attrs then
      for cls, _ in pairs(THEOREM_CLASSES) do
        if attrs:match("%." .. cls .. "%f[%s%}#.]") or attrs:match("%." .. cls .. "$") then
          local noinc = attrs:match("%.noinc") ~= nil
          local label = attrs:match("#([%w%-_]+)")
          table.insert(entries, {label = label, class = cls, noinc = noinc})
          if not noinc then
            count = count + 1
          end
          if label then
            table.insert(current_slide_labels, label)
          end
          break
        end
      end
    else
      local cls = line:match("^:::%s+(%a+)%s*$")
      if cls and THEOREM_CLASSES[cls] then
        table.insert(entries, {label = nil, class = cls, noinc = false})
        count = count + 1
      end
    end
  end

  -- Don't forget the last slide's labels
  if #current_slide_labels > 0 then
    table.insert(slide_groups, current_slide_labels)
  end

  return count, entries, slide_groups
end

-- Scan all *-deck.md files and compute the numbering scheme.
-- `doc` is the current Pandoc document (used to identify the current file via metadata).
local function build_numbering_scheme(doc)
  local deck_dir = get_deck_dir()

  -- Get the current file's subtitle from document metadata (works with decker)
  local current_subtitle = doc.meta.subtitle and pandoc.utils.stringify(doc.meta.subtitle) or nil
  -- Also try PANDOC_STATE.input_files for standalone pandoc usage
  local current_file_from_state = PANDOC_STATE.input_files[1]
    and PANDOC_STATE.input_files[1]:match("([^/]+)$") or nil

  -- List all *-deck.md files
  local filenames = {}
  local handle = io.popen('ls "' .. deck_dir .. '"*-deck.md 2>/dev/null')
  if handle then
    for path in handle:lines() do
      local basename = path:match("([^/]+)$")
      table.insert(filenames, {path = path, basename = basename})
    end
    handle:close()
  end

  -- Sort by basename (gives us 00, 01, 02, ..., 13 ordering)
  table.sort(filenames, function(a, b) return a.basename < b.basename end)

  -- For each file: read, parse chapter + subtitle, count theorems
  local chapter_files = {}  -- chapter → ordered list of basenames
  local file_data = {}      -- basename → {chapter, count, entries, subtitle}

  for _, f in ipairs(filenames) do
    local fh = io.open(f.path, "r")
    if fh then
      local text = fh:read("*a")
      fh:close()

      local chapter, subtitle = parse_frontmatter(text)
      if chapter then
        local count, entries, slide_groups = count_theorems_raw(text)
        file_data[f.basename] = {
          chapter = chapter,
          count = count,
          entries = entries,
          subtitle = subtitle,
          slide_groups = slide_groups,
        }
        if not chapter_files[chapter] then
          chapter_files[chapter] = {}
        end
        table.insert(chapter_files[chapter], f.basename)
      end
    end
  end

  -- Compute offset for each file = 1 + sum of theorem counts in earlier files of same chapter
  local file_offsets = {}
  for ch, files in pairs(chapter_files) do
    local cumulative = 1
    for _, basename in ipairs(files) do
      file_offsets[basename] = cumulative
      cumulative = cumulative + (file_data[basename].count or 0)
    end
  end

  -- Build label → section_target mapping from slide groups.
  -- The first labeled theorem on each slide gets promoted to the <section> id;
  -- all other labels on that slide navigate to the same section.
  local label_to_section = {}
  for _, data in pairs(file_data) do
    for _, group in ipairs(data.slide_groups) do
      local first = group[1]
      for _, lbl in ipairs(group) do
        label_to_section[lbl] = first
      end
    end
  end

  -- Build label map: label → {display_name, chapter, number, file_html, section_target}
  local label_map = {}
  for ch, files in pairs(chapter_files) do
    for _, basename in ipairs(files) do
      local data = file_data[basename]
      local num = file_offsets[basename]
      for _, entry in ipairs(data.entries) do
        if not entry.noinc then
          if entry.label then
            label_map[entry.label] = {
              display = DISPLAY_NAMES[entry.class],
              chapter = data.chapter,
              number = num,
              file = basename:gsub("%.md$", ".html"),
              section_target = label_to_section[entry.label] or entry.label,
            }
          end
          num = num + 1
        end
      end
    end
  end

  -- Identify the current file:
  -- 1. Try matching by input_files basename (standalone pandoc)
  -- 2. Try matching by subtitle (decker — feeds via stdin, no input_files)
  local current_file = nil
  if current_file_from_state and file_data[current_file_from_state] then
    current_file = current_file_from_state
  elseif current_subtitle then
    for basename, data in pairs(file_data) do
      if data.subtitle == current_subtitle then
        current_file = basename
        break
      end
    end
  end

  local current_data = current_file and file_data[current_file]
  local current_offset = current_file and file_offsets[current_file]
  local current_chapter = current_data and current_data.chapter
    or (doc.meta.chapter and tonumber(pandoc.utils.stringify(doc.meta.chapter)))
    or 0

  return {
    label_map = label_map,
    current_offset = current_offset or 1,
    current_chapter = current_chapter,
    current_file = current_file,
  }
end

--------------------------------------------------------------------------------
-- Phase 2: AST transformation
--------------------------------------------------------------------------------

local scheme  -- will be populated in the Pandoc filter

function Pandoc(doc)
  scheme = build_numbering_scheme(doc)

  local enumeration = scheme.current_offset
  local chapter = scheme.current_chapter

  -- Walk all Div elements
  local function process_div(div)
    local cls = get_theorem_class(div)
    if not cls then return nil end

    local is_inline = div.classes:includes("inline")
    local is_noinc = div.classes:includes("noinc")

    -- Remove helper classes from output
    if is_inline then
      div.classes = div.classes:filter(function(c) return c ~= "inline" end)
    end
    if is_noinc then
      div.classes = div.classes:filter(function(c) return c ~= "noinc" end)
    end

    if is_noinc then
      enumeration = enumeration - 1
    end

    local display = DISPLAY_NAMES[cls]
    local label_text = display .. " " .. chapter .. "." .. enumeration

    -- Check for an H2 header inside the div → append its text, then remove it
    local header_text = nil
    local new_content = {}
    for _, block in ipairs(div.content) do
      if block.t == "Header" and block.level == 2 then
        header_text = pandoc.utils.stringify(block)
      else
        table.insert(new_content, block)
      end
    end
    if header_text then
      label_text = label_text .. " " .. header_text
      div.content = new_content
    end

    -- Set the div's ID if it has a label
    -- (Pandoc preserves {#id} from the fenced div syntax, so div.identifier may already be set)

    -- Create the bold label
    local bold_label
    if is_inline then
      bold_label = pandoc.Strong(pandoc.Str(label_text .. ". "))
    else
      bold_label = pandoc.Strong(pandoc.Str(label_text))
    end

    -- Prepend the label
    if is_inline then
      -- Prepend to the first Para/Plain block
      local done = false
      for i, block in ipairs(div.content) do
        if (block.t == "Para" or block.t == "Plain") and not done then
          local new_inlines = {bold_label}
          for _, inline in ipairs(block.content) do
            table.insert(new_inlines, inline)
          end
          if block.t == "Para" then
            div.content[i] = pandoc.Para(new_inlines)
          else
            div.content[i] = pandoc.Plain(new_inlines)
          end
          done = true
        end
      end
      if not done then
        table.insert(div.content, 1, pandoc.Para({bold_label}))
      end
    else
      table.insert(div.content, 1, pandoc.Para({bold_label}))
    end

    enumeration = enumeration + 1
    return div
  end

  -- Walk all Link elements to resolve references
  local function process_link(link)
    if #link.content > 0 then return nil end  -- only fill empty links

    local target = link.target
    -- Extract the fragment (label)
    local label = target:match("#([%w%-_]+)$")
    if not label then return nil end

    local entry = scheme.label_map[label]
    if not entry then
      io.stderr:write("WARNING [theorem-numbering.lua]: unknown label '" .. label .. "'\n")
      return nil
    end

    local display_text = entry.display .. " " .. entry.chapter .. "." .. entry.number
    link.content = {pandoc.Str(display_text)}

    -- Build target with #/ prefix so Reveal.js navigates by section id.
    -- Use section_target (the first label on the slide) which gets promoted
    -- to the <section> element — this handles multiple definitions per slide.
    local nav_target = entry.section_target or label
    local current_html = scheme.current_file and scheme.current_file:gsub("%.md$", ".html") or ""
    local target_file = target:match("^(.-)#") or ""
    if target_file == "" or target_file == current_html then
      link.target = "#/" .. nav_target
    else
      link.target = target_file .. "#/" .. nav_target
    end

    return link
  end

  -- Apply transformations
  doc = doc:walk({Div = process_div})
  doc = doc:walk({Link = process_link})

  -- Phase 3: Promote labeled theorem-div IDs to slide-level Divs.
  -- Decker wraps each slide in a Div with classes={slide,level1}.
  -- Reveal.js navigates by <section id="...">; Pandoc turns these Div ids
  -- into section ids.  By moving the label up, #/label URLs work natively.
  local function find_first_labeled_theorem(block)
    if block.t == "Div" then
      if get_theorem_class(block) and block.identifier ~= "" then
        return block
      end
      for _, child in ipairs(block.content) do
        local found = find_first_labeled_theorem(child)
        if found then return found end
      end
    end
    return nil
  end

  for i, block in ipairs(doc.blocks) do
    if block.t == "Div" and block.classes:includes("slide") then
      local labeled_div = find_first_labeled_theorem(block)
      if labeled_div then
        block.identifier = labeled_div.identifier
        labeled_div.identifier = ""
      end
    end
  end

  return doc
end
