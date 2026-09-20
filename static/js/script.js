/**
 * EcoPulse AI — Main JavaScript
 * Handles minor UI enhancements (Gemini fetch is inline in result.html)
 */

document.addEventListener('DOMContentLoaded', () => {
  // ---- Active form validation feedback ----
  const numericInputs = document.querySelectorAll('input[type="number"]');
  numericInputs.forEach(input => {
    input.addEventListener('input', () => {
      const val = parseFloat(input.value);
      if (input.value && isNaN(val)) {
        input.style.borderColor = '#E53935';
      } else {
        input.style.borderColor = '';
      }
    });
  });

  // ---- Smooth scroll for anchor links ----
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

  // ---- Auto-fade alert messages after 8 seconds ----
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(alert => {
    setTimeout(() => {
      alert.style.transition = 'opacity 0.6s';
      alert.style.opacity = '0';
      setTimeout(() => alert.remove(), 700);
    }, 8000);
  });

  // ---- Highlight table row on hover ----
  const tableRows = document.querySelectorAll('.comparison-table tbody tr');
  tableRows.forEach(row => {
    row.addEventListener('mouseenter', () => {
      if (!row.classList.contains('best-row')) {
        row.style.backgroundColor = '#F8F8FC';
      }
    });
    row.addEventListener('mouseleave', () => {
      if (!row.classList.contains('best-row')) {
        row.style.backgroundColor = '';
      }
    });
  });

  // ---- Date/time field: set default to now ----
  const dateInput = document.getElementById('date_input');
  if (dateInput && !dateInput.value) {
    const now = new Date();
    // Format: YYYY-MM-DDTHH:MM
    const pad = n => String(n).padStart(2, '0');
    const localStr = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}`;
    dateInput.value = localStr;
  }
});
