<script lang="ts">
  type Props = { name: string; size?: number }

  let { name, size = 18 }: Props = $props()
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.7"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
  focusable="false"
>
  {#if name === 'mic'}
    <rect x="9" y="2.5" width="6" height="11" rx="3" />
    <path d="M5.5 11a6.5 6.5 0 0 0 13 0" />
    <path d="M12 17.5V21" />
  {:else if name === 'stop'}
    <rect x="6.5" y="6.5" width="11" height="11" rx="2.5" fill="currentColor" stroke="none" />
  {:else if name === 'file-text'}
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" />
    <path d="M9 13h6M9 17h4" />
  {:else if name === 'file-audio'}
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" />
    <path d="M9.5 16v-2M12 17.5v-5M14.5 16v-3" />
  {:else if name === 'clock'}
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 7.5V12l3 2" />
  {:else if name === 'layers'}
    <path d="m12 3.5 8 4.5-8 4.5-8-4.5z" />
    <path d="m4 12.5 8 4.5 8-4.5" />
    <path d="m4 17 8 4.5 8-4.5" />
  {:else if name === 'upload'}
    <path d="M12 16.5V4.5" />
    <path d="m7.5 9 4.5-4.5L16.5 9" />
    <path d="M4 20h16" />
  {:else if name === 'download'}
    <path d="M12 4v12" />
    <path d="m7.5 11.5 4.5 4.5 4.5-4.5" />
    <path d="M4 20h16" />
  {:else if name === 'copy'}
    <rect x="9" y="9" width="11" height="11" rx="2.5" />
    <path d="M15 5.5A1.5 1.5 0 0 0 13.5 4H6a2 2 0 0 0-2 2v7.5A1.5 1.5 0 0 0 5.5 15" />
  {:else if name === 'check'}
    <path d="m5 12.5 4.5 4.5L19 7" />
  {:else if name === 'trash'}
    <path d="M4 7h16" />
    <path d="M9.5 7V5.5A1.5 1.5 0 0 1 11 4h2a1.5 1.5 0 0 1 1.5 1.5V7" />
    <path d="M6.5 7l.8 12a1.5 1.5 0 0 0 1.5 1.4h6.4a1.5 1.5 0 0 0 1.5-1.4l.8-12" />
  {:else if name === 'chevron'}
    <path d="m6.5 9.5 5.5 5.5 5.5-5.5" />
  {:else if name === 'swap'}
    <path d="M4 8.5h13" />
    <path d="m14 5 3.5 3.5L14 12" />
    <path d="M20 15.5H7" />
    <path d="M10 12l-3.5 3.5L10 19" />
  {:else if name === 'activity'}
    <path d="M3 12h3.2l2.3-6 4.2 12 2.3-6H21" />
  {:else if name === 'sparkle'}
    <path d="m12 4 1.9 5.6L19.5 11.5l-5.6 1.9L12 19l-1.9-5.6L4.5 11.5l5.6-1.9z" />
  {:else if name === 'alert'}
    <path d="M12 4.5 21 19.5H3z" />
    <path d="M12 10v4" />
    <path d="M12 17h.01" />
  {:else if name === 'sliders'}
    <path d="M4 7h9M17 7h3" />
    <circle cx="15" cy="7" r="2" />
    <path d="M4 17h3M11 17h9" />
    <circle cx="9" cy="17" r="2" />
  {:else if name === 'reset'}
    <path d="M4 12a8 8 0 1 0 2.6-5.9" />
    <path d="M4 4.5v4h4" />
  {:else if name === 'wave'}
    <path d="M4 11v2M8 7.5v9M12 4.5v15M16 8.5v7M20 11v2" />
  {:else if name === 'x'}
    <path d="m6.5 6.5 11 11M17.5 6.5l-11 11" />
  {/if}
</svg>
