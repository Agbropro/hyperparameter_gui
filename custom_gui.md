# Beginner's guide to customizing the GUI

You do not need to understand JavaScript to change the name, wording, colors, or most of the layout. The frontend has only three files:

| File | What it controls | Edit it when you want to… |
| --- | --- | --- |
| `frontend/index.html` | Page content and structure | Change names, labels, descriptions, and sections |
| `frontend/styles.css` | Appearance | Change colors, spacing, fonts, sizes, and shapes |
| `frontend/app.js` | Behavior and backend data | Change dynamic model choices, submission behavior, or result rendering |

The separate final-training and validation pages use these additional files:

| Page | Content and appearance | Behavior |
| --- | --- | --- |
| Train best | `frontend/training.html`, `frontend/training.css` | `frontend/training.js` |
| Validate | `frontend/validation.html`, `frontend/validation.css` | `frontend/validation.js` |

For simple visual customization, you will mostly edit `index.html` and `styles.css`. Avoid changing IDs such as `id="experiment-form"`; JavaScript uses those IDs to find elements.

## Change the title shown in the screenshot

Open `frontend/index.html` and find:

```html
<a class="brand" href="/"><span class="mark">Y</span><span>YOLO <b>Studio</b></span></a>
```

It has two visible pieces:

- `<span class="mark">Y</span>` is the **Y inside the black square**.
- `<span>YOLO <b>Studio</b></span>` is the **YOLO Studio name**. Text inside `<b>` is bold.

For example, to show an **H** icon and the name **Hyper Tune**, change it to:

```html
<a class="brand" href="/"><span class="mark">H</span><span>Hyper <b>Tune</b></span></a>
```

The browser-tab title is separate. Find:

```html
<title>YOLO Hyperparameter Studio</title>
```

Change the words between `<title>` and `</title>`.

## Change other text

All the main static text is in `frontend/index.html`. Change only the text between HTML tags and leave the tags intact.

Example:

```html
<h1>Build a better model,<br /><em>one trial at a time.</em></h1>
```

can become:

```html
<h1>Find your best model,<br /><em>automatically.</em></h1>
```

Useful text to search for:

- `Local training engine` — top-right status text
- `HYPERPARAMETER SEARCH / 01` — small text above the headline
- `Build a better model` — large headline
- `New experiment` — left panel title
- `Experiments` — right panel title
- `Start optimization` — submit button

Some text is created dynamically by `frontend/app.js`, including experiment cards, the details window, error messages, version notes, and metric buttons. Use your editor's search function to find the exact wording before changing it.

## Change the colors

Open `frontend/styles.css`. The first lines define the main colors in one place:

```css
:root {
  --ink: #12130f;
  --paper: #f3f1e9;
  --card: rgba(255,255,255,.76);
  --line: rgba(18,19,15,.13);
  --acid: #d9ff43;
  --orange: #ff714b;
  --purple: #795cff;
  --muted: #727369;
  --radius: 18px;
}
```

The current file puts several variables on each line, but you can safely format them like the example above. The variables mean:

- `--ink`: main text, dark buttons, and the logo square
- `--paper`: page background
- `--card`: panel background
- `--line`: subtle borders
- `--acid`: green highlight and logo letter
- `--orange`: reserved accent color
- `--purple`: headline accent, main button, and progress bars
- `--muted`: secondary text
- `--radius`: panel corner roundness

Colors beginning with `#` are hex colors. You can use any online color picker to obtain them. After changing a variable, every style that uses `var(--variable-name)` updates automatically.

### Example blue theme

```css
:root {
  --ink: #111827;
  --paper: #eff6ff;
  --card: rgba(255,255,255,.82);
  --line: rgba(17,24,39,.13);
  --acid: #67e8f9;
  --orange: #fb923c;
  --purple: #2563eb;
  --muted: #64748b;
  --radius: 18px;
}
```

## Customize the logo square

The `.mark` style in `frontend/styles.css` controls the square logo:

```css
.mark {
  width: 31px;
  height: 31px;
  border-radius: 9px;
  background: var(--ink);
  color: var(--acid);
  transform: rotate(-5deg);
}
```

- Increase `width` and `height` to make it larger.
- Change `border-radius` to `50%` to make it a circle.
- Change `transform: rotate(-5deg)` to `transform: none` to make it straight.
- `background` controls the square color.
- `color` controls the letter color.

To use an image instead of a letter:

1. Put an image such as `logo.png` inside the `frontend` folder.
2. Replace `<span class="mark">Y</span>` in `index.html` with:

```html
<img class="brand-logo" src="/assets/logo.png" alt="My logo" />
```

3. Add this to `styles.css`:

```css
.brand-logo {
  width: 36px;
  height: 36px;
  object-fit: contain;
}
```

## Change fonts

The page currently downloads **Manrope** and **DM Mono** from Google Fonts. The font link is near the top of `frontend/index.html`:

```html
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
```

The main font is set in `frontend/styles.css`:

```css
body { font-family: Manrope, sans-serif; }
```

To use Arial without downloading a font, change it to:

```css
body { font-family: Arial, sans-serif; }
```

You can then remove the three Google Fonts `<link>` lines if you want the frontend to work without internet access.

## Change spacing and panel shape

Common controls in `frontend/styles.css`:

```css
.topbar { padding: 26px 42px; }
main { padding: 45px 42px 80px; }
.config-panel { padding: 29px; }
.results-panel { padding: 29px; }
```

CSS padding values work clockwise:

```text
padding: top right bottom left;
```

When there are only two values, the first is top/bottom and the second is left/right.

To make corners less rounded, change `--radius: 18px` to something like `--radius: 8px`. To make them more rounded, try `24px`.

## Change the two-column layout

This rule controls the form and experiment-list columns:

```css
.workspace {
  grid-template-columns: minmax(600px, 1.55fr) minmax(360px, .85fr);
}
```

The first number is the form column and the second is the results column. For equal columns, use:

```css
grid-template-columns: 1fr 1fr;
```

The `@media` rules at the bottom make the layout stack vertically on smaller screens. Keep those rules unless you intentionally want to change mobile behavior.

## HTML safety rules for beginners

HTML normally has an opening and closing tag:

```html
<span>Visible text</span>
```

Keep these rules in mind:

1. Change text between tags freely.
2. Keep matching opening and closing tags.
3. Keep `id`, `name`, `value`, and `data-*` attributes unless you also update the JavaScript.
4. Keep input types such as `type="number"` because they control validation.
5. Make one small change at a time and refresh the browser.

## What JavaScript does in this project

`frontend/app.js` connects the page to the Python API. It:

- loads YOLO versions, sizes, metrics, and parameter ranges;
- builds the correct model filename;
- validates the dataset path through the API;
- submits experiments;
- refreshes running experiment progress;
- shows trial results and handles cancellation.

You do not need to edit JavaScript for normal styling. If you do edit it, punctuation matters: quotes, parentheses, braces, commas, and semicolons all have meaning.

You can check the file for syntax errors with:

```bash
node --check frontend/app.js
```

## Preview your changes

Run the application from the project folder:

```bash
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. After editing HTML or CSS, refresh the browser. If an old style remains, perform a hard refresh:

- Windows/Linux: `Ctrl + Shift + R`
- macOS: `Command + Shift + R`

`--reload` restarts Python when backend files change. HTML, CSS, and JavaScript usually only need a browser refresh.

## Recover from a mistake

Before a large redesign, copy the three frontend files somewhere safe or commit them with Git. If the page breaks:

1. Undo the most recent edit.
2. Check that HTML tags still have their closing tags.
3. Open the browser developer tools with `F12` and look at the **Console** for red errors.
4. Run `node --check frontend/app.js` if you edited JavaScript.
5. Restart Uvicorn and hard-refresh the page.

For your first customization, change only the brand text, browser title, headline, and the colors under `:root`. Those changes are low risk and do not affect model training.
