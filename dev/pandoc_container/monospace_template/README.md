This folder contains css and a pandoc template to use for generating monospace documents.

Note: When importing mermaid in the template, v10 includes an elk backend but v11 doesn't. Therefore, they need to be imported differently:

For v10:
```
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.mjs';
    mermaid.initialize({startOnLoad: true})
</script>
```
For v11:
```
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.mjs';
    import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0.1.4/dist/mermaid-layout-elk.esm.min.mjs';
    mermaid.registerLayoutLoaders(elkLayouts);
    mermaid.initialize({startOnLoad: true})
</script>
```

There also seems to be some differences in how the configs work, e.g. in v11 specifying elk in the inline yaml looks like it works fine, but doesn't seem to work in v10. 