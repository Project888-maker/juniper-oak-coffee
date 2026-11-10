// Set this to a form endpoint (Formspree, your own API, etc.) to start
// collecting waitlist emails. While empty, the form tells the visitor it is
// not connected instead of pretending the email was saved.
const WAITLIST_ENDPOINT = "";

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Nav background after scroll
const nav = document.getElementById("nav");
const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 24);
onScroll();
window.addEventListener("scroll", onScroll, { passive: true });

// Scroll reveal
const revealEls = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window && !reduceMotion) {
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        io.unobserve(entry.target);
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );
  revealEls.forEach((el, i) => {
    el.style.transitionDelay = `${(i % 3) * 80}ms`;
    io.observe(el);
  });
} else {
  revealEls.forEach((el) => el.classList.add("in"));
}

// Hero video: only load and play when motion is welcome; the poster image
// stays underneath if the file is missing or autoplay is blocked.
const video = document.getElementById("heroVideo");
if (video && !reduceMotion) {
  video.addEventListener("playing", () => video.classList.add("on"), { once: true });
  video.preload = "auto";
  video.play().catch(() => {});
}

// Only one programme module open at a time
const modules = document.querySelectorAll("#programme details");
modules.forEach((d) =>
  d.addEventListener("toggle", () => {
    if (d.open) modules.forEach((other) => other !== d && (other.open = false));
  })
);

// Checkout modal
const modal = document.getElementById("checkout");
const planLabel = document.getElementById("checkoutPlan");
const form = document.getElementById("waitlist");
const msg = document.getElementById("formMsg");
let chosenPlan = "";

document.querySelectorAll("[data-plan]").forEach((btn) =>
  btn.addEventListener("click", () => {
    chosenPlan = btn.dataset.plan;
    planLabel.textContent = chosenPlan;
    msg.textContent = "";
    modal.showModal();
  })
);
modal.addEventListener("click", (e) => {
  if (e.target === modal) modal.close();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = form.email.value.trim();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    msg.textContent = "Please enter a valid email address.";
    form.email.focus();
    return;
  }
  if (!WAITLIST_ENDPOINT) {
    msg.textContent = "The waitlist isn't connected yet, so your email was not saved. Please check back soon.";
    return;
  }
  msg.textContent = "Sending…";
  try {
    const res = await fetch(WAITLIST_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ email, plan: chosenPlan }),
    });
    if (!res.ok) throw new Error(res.statusText);
    form.reset();
    msg.textContent = "You're on the list. We'll email you when enrolment opens.";
  } catch {
    msg.textContent = "That didn't go through. Please try again in a moment.";
  }
});

document.getElementById("year").textContent = new Date().getFullYear();
