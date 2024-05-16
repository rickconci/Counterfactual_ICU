import wandb
from utils_6 import interpolate_colors
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def plot_trajectories(X, Y, Y_hat, Y_cf, Y_hat_cf, latent_traj, chart_type="val"):
        # Ensure tensors are on CPU for plotting
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        Y_hat = Y_hat.cpu().numpy()
        Y_cf = Y_cf.cpu().numpy()
        Y_hat_cf = Y_hat_cf.cpu().numpy()
        latent_traj = latent_traj.cpu().numpy()

        # Only take the first item of the batch for simplicity
        X = X[0]
        Y = Y[0]
        Y_hat = Y_hat[0]
        Y_cf = Y_cf[0]
        Y_hat_cf = Y_hat_cf[0]
        latent_traj = latent_traj[0]

        plot_latent_trajectories(latent_traj, chart_type)
        plot_factual_predicted_trajectories( X, Y, Y_hat, chart_type)
        plot_counterfactual_predicted_trajectories(X, Y_cf, Y_hat_cf, chart_type)



def plot_latent_trajectories(latent_traj, chart_type="val"):
    start_color = (0,255,127)  # Royal blue
    end_color = (238,130,238)     # Dark orange
    
    latent_colors = interpolate_colors(start_color, end_color, latent_traj.shape[2])

    fig = go.Figure()
    time_latent = np.arange(latent_traj.shape[1])
    for i in range(latent_traj.shape[2]):
        mean_latent = np.mean(latent_traj[:, :, i], axis=0)
        std_latent = np.std(latent_traj[:, :, i], axis=0)
        upper_bound = mean_latent + std_latent
        lower_bound = mean_latent - std_latent
        
        r, g, b = latent_colors[i]
        main_color = f'rgb({r}, {g}, {b})'
        light_color_rgba = f'rgba({r}, {g}, {b}, 0.2)'


        fig.add_trace(
            go.Scatter(x=time_latent, y=mean_latent, mode='lines', name=f'Latent Dim {i}', line=dict(color=main_color))
        )
        fig.add_trace(
            go.Scatter(x=time_latent, y=upper_bound, line=dict(color=light_color_rgba), showlegend=False)
        )
        fig.add_trace(
            go.Scatter(x=time_latent, y=lower_bound, line=dict(color=light_color_rgba), fill='tonexty', showlegend=False)
        )

    fig.update_layout(height=600, width=800, title_text=f"Latent Trajectories")
    fig.update_xaxes(title_text="Time Steps")
    fig.update_yaxes(title_text="Values")

    wandb.log({"Latent SDE trajectories": fig})

def plot_factual_predicted_trajectories(X, Y, Y_hat, chart_type="val"):
    base_colors = ['dodgerblue', 'sandybrown', 'mediumseagreen', 'mediumorchid', 'coral', 'slategray']
    light_colors = ['lightblue', 'peachpuff', 'lightgreen', 'plum', 'lightsalmon', 'lightgray']
    dark_colors = ['darkblue', 'darkorange', 'darkgreen', 'darkorchid', 'darkred', 'darkslategray']

    fig = go.Figure()
    time_x = np.arange(X.shape[0])
    time_y = time_x[-1] + 1 + np.arange(Y.shape[0])

    for i in range(max(X.shape[1], Y.shape[1])):  # Handle different dimensions for X and Y/Y_hat
        if i < Y_hat.shape[2]:  # Ensure we do not go out of index for Y_hat
            mean_y_hat = np.mean(Y_hat[:, :, i], axis=0)
            std_y_hat = np.std(Y_hat[:, :, i], axis=0)
            upper_bound = mean_y_hat + std_y_hat
            lower_bound = mean_y_hat - std_y_hat

        if i < X.shape[1]:
            fig.add_trace(
                go.Scatter(x=time_x, y=X[:, i], mode='lines', name=f'Input_{i}', line=dict(color=base_colors[i % len(base_colors)]))
            )
        if i < Y.shape[1]:
            fig.add_trace(
                go.Scatter(x=time_y, y=Y[:, i], mode='lines', name=f'Factual_{i}', line=dict(color=base_colors[i % len(base_colors)]))
            )
            if i < Y_hat.shape[2]:  # Check again to ensure Y_hat has this index
                fig.add_trace(
                    go.Scatter(x=time_y, y=mean_y_hat, mode='lines', name=f'Predicted_{i}', line=dict(color=dark_colors[i % len(dark_colors)]))
                )
                fig.add_trace(
                    go.Scatter(x=time_y, y=upper_bound, line=dict(color=light_colors[i % len(light_colors)]), showlegend=False)
                )
                fig.add_trace(
                    go.Scatter(x=time_y, y=lower_bound, line=dict(color=light_colors[i % len(light_colors)]), fill='tonexty', showlegend=False)
                )

    fig.update_layout(height=600, width=800, title_text=f"Observed & Predicted Trajectories")
    fig.update_xaxes(title_text="Time Steps")
    fig.update_yaxes(title_text="Values")
    wandb.log({"Factual observed & predicted": fig})

def plot_counterfactual_predicted_trajectories(X, Y_cf, Y_hat_cf, chart_type="val"):
    base_colors = ['dodgerblue', 'sandybrown', 'mediumseagreen', 'mediumorchid', 'coral', 'slategray']
    light_colors = ['lightblue', 'peachpuff', 'lightgreen', 'plum', 'lightsalmon', 'lightgray']
    dark_colors = ['darkblue', 'darkorange', 'darkgreen', 'darkorchid', 'darkred', 'darkslategray']

    fig = go.Figure()
    time_x = np.arange(X.shape[0])
    time_y = time_x[-1] + 1 + np.arange(Y_cf.shape[0])

    for i in range(max(X.shape[1], Y_cf.shape[1])):  # Handle different dimensions for X and Y/Y_hat
        if i < Y_hat_cf.shape[2]:  # Ensure we do not go out of index for Y_hat
            mean_y_hat = np.mean(Y_hat_cf[:, :, i], axis=0)
            std_y_hat = np.std(Y_hat_cf[:, :, i], axis=0)
            upper_bound = mean_y_hat + std_y_hat
            lower_bound = mean_y_hat - std_y_hat

        if i < X.shape[1]:
            fig.add_trace(
                go.Scatter(x=time_x, y=X[:, i], mode='lines', name=f'Input_{i}', line=dict(color=base_colors[i % len(base_colors)]))
            )
        if i < Y_cf.shape[1]:
            fig.add_trace(
                go.Scatter(x=time_y, y=Y_cf[:, i], mode='lines', name=f'Counterfactual{i}', line=dict(color=base_colors[i % len(base_colors)]))
            )
            if i < Y_hat_cf.shape[2]:  # Check again to ensure Y_hat has this index
                fig.add_trace(
                    go.Scatter(x=time_y, y=mean_y_hat, mode='lines', name=f'Predicted_Counterfactual_{i}', line=dict(color=dark_colors[i % len(dark_colors)]))
                )
                fig.add_trace(
                    go.Scatter(x=time_y, y=upper_bound, line=dict(color=light_colors[i % len(light_colors)]), showlegend=False)
                )
                fig.add_trace(
                    go.Scatter(x=time_y, y=lower_bound, line=dict(color=light_colors[i % len(light_colors)]), fill='tonexty', showlegend=False)
                )

    fig.update_layout(height=600, width=800, title_text=f"Counterfactual 'observed' & predicted")
    fig.update_xaxes(title_text="Time Steps")
    fig.update_yaxes(title_text="Values")

    wandb.log({"Counterfactual 'observed' & predicted": fig})
    



def plot_trajectories_old(X, Y, Y_hat, latent_traj, chart_type):
    # Ensure tensors are on CPU for plotting
    X = X.cpu()
    Y = Y.cpu()
    Y_hat = Y_hat.cpu()
    latent_traj = latent_traj.cpu()  # Assume latent_traj is passed or correctly fetched earlier

    # Only take the first item of the batch
    X = X[0]
    Y = Y[0]
    Y_hat = Y_hat[0]
    latent_traj = latent_traj[0]

    # Define a color palette that provides distinct colors for multiple dimensions
    base_colors = ['dodgerblue', 'sandybrown', 'mediumseagreen', 'mediumorchid', 'coral', 'slategray']
    light_colors = ['lightblue', 'peachpuff', 'lightgreen', 'plum', 'lightsalmon', 'lightgray']
    dark_colors = ['darkblue', 'darkorange', 'darkgreen', 'darkorchid', 'darkred', 'darkslategray']

    start_color = (0,255,127)  # Royal blue
    end_color = (238,130,238)     # Dark orange
    latent_colors = interpolate_colors(start_color, end_color, latent_traj.shape[1], rgb=True)


    fig = make_subplots(rows=2, cols=1, subplot_titles=("Latent Trajectories", "Observed Trajectories"))

    # Time axes for observed and latent data
    time_x = np.arange(X.shape[0])
    time_y = time_x[-1] + 1 + np.arange(Y.shape[0])
    time_latent = np.arange(latent_traj.shape[0])

    # Plot latent trajectories
    for i in range(latent_traj.shape[1]):  # Assuming second dim is the feature dimension of latent space
        fig.add_trace(
            go.Scatter(x=time_latent, y=latent_traj[:, i], mode='lines', name=f'Latent Dim {i}', line=dict(color=latent_colors[i])),
            row=1, col=1
        )

    # Plot X, Y, and Y_hat trajectories
    for i in range(max(X.shape[1], Y.shape[1])):  # Handle different dimensions for X and Y/Y_hat
        if i < X.shape[1]:
            fig.add_trace(
                go.Scatter(x=time_x, y=X[:, i], mode='lines', name=f'Input_{i}', line=dict(color=base_colors[i % len(base_colors)])),
                row=2, col=1
            )
        if i < Y.shape[1]:
            fig.add_trace(
                go.Scatter(x=time_y, y=Y[:, i], mode='lines', name=f'Factual_{i}', line=dict(color=light_colors[i % len(light_colors)])),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_y, y=Y_hat[:, i], mode='lines', name=f'Predicted_{i}', line=dict(color=dark_colors[i % len(dark_colors)])),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(height=600, width=800, title_text=f"{chart_type} Validation Trajectories")
    fig.update_xaxes(title_text="Time Steps")
    fig.update_yaxes(title_text="Values", row=1, col=1)
    fig.update_yaxes(title_text="Values", row=2, col=1)

    # Optionally show the figure, useful for interactive sessions or debugging
    # fig.show()

    # Log the figure to wandb
    wandb.log({"predictions_vs_actuals": fig})

