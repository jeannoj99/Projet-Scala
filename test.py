my_div = html.Div([
    dash_table.DataTable(
        id='my-table',
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        # Tous tes styles personnalisés ici
    )
])


def export_div_simple(div_component, filename="export.html"):
    """Version simplifiée qui utilise la serialization Dash"""
    import json
    
    # Sérialiser le composant
    component_dict = div_component.to_plotly_json()
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DataTable Export</title>
    <link rel="stylesheet" href="https://unpkg.com/dash-table@5.0.0/dash_table/bundle.css">
</head>
<body>
    <div id="react-entry-point"></div>
    
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/dash-table@5.0.0/dash_table/bundle.js"></script>
    
    <script>
        const component = {json.dumps(component_dict)};
        
        // Rendu du composant
        const container = document.getElementById('react-entry-point');
        ReactDOM.render(
            React.createElement(window.dash_table.DataTable, component.props),
            container
        );
    </script>
</body>
</html>"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename