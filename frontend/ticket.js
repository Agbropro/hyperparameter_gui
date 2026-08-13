(() => {
  const opener = document.querySelector("[data-ticket-open]");
  if (!opener) return;

  document.body.insertAdjacentHTML("beforeend", `
    <div class="ticket-modal" data-ticket-modal hidden>
      <div class="ticket-backdrop" data-ticket-close></div>
      <section class="ticket-card" role="dialog" aria-modal="true" aria-labelledby="ticket-heading">
        <button class="ticket-close" type="button" data-ticket-close aria-label="Close ticket form">×</button>
        <h2 id="ticket-heading">Send a ticket</h2>
        <p>Report a bug, suggest a feature, or leave another note for the developer.</p>
        <form class="ticket-form" data-ticket-form>
          <label><span>Ticket title</span><input name="title" minlength="3" maxlength="120" required placeholder="Short summary" /></label>
          <label><span>Type</span><select name="type"><option value="feature">Feature</option><option value="bug">Bug</option><option value="misc">Miscellaneous</option></select></label>
          <label><span>Description</span><textarea name="message" minlength="5" maxlength="5000" required placeholder="What happened, or what would you like to see?"></textarea></label>
          <p class="ticket-feedback" data-ticket-feedback hidden></p>
          <button class="ticket-submit" type="submit">Send ticket</button>
        </form>
      </section>
    </div>`);

  const modal = document.querySelector("[data-ticket-modal]");
  const form = document.querySelector("[data-ticket-form]");
  const feedback = document.querySelector("[data-ticket-feedback]");
  const close = () => { modal.hidden = true; document.body.style.overflow = ""; };
  const open = () => { feedback.hidden = true; modal.hidden = false; document.body.style.overflow = "hidden"; form.elements.title.focus(); };

  opener.addEventListener("click", open);
  modal.querySelectorAll("[data-ticket-close]").forEach(button => button.addEventListener("click", close));
  document.addEventListener("keydown", event => { if (event.key === "Escape" && !modal.hidden) close(); });
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const submit = form.querySelector("button[type=submit]");
    const data = new FormData(form);
    submit.disabled = true;
    feedback.hidden = true;
    try {
      const response = await fetch("/api/tickets", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: data.get("title"), type: data.get("type"), message: data.get("message"), page: window.location.pathname })
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
      form.reset();
      feedback.textContent = `Ticket sent. Reference: ${body.id}`;
      feedback.className = "ticket-feedback";
      feedback.hidden = false;
    } catch (error) {
      feedback.textContent = error.message;
      feedback.className = "ticket-feedback error";
      feedback.hidden = false;
    } finally { submit.disabled = false; }
  });
})();
