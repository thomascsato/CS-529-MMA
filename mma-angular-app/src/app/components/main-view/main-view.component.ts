import { Component } from '@angular/core';

@Component({
  selector: 'app-main-view',
  templateUrl: './main-view.component.html',
  styleUrl: './main-view.component.css'
})
export class MainViewComponent {
  onButtonClick() {
    console.log('Button clicked!');
    // Add your button click logic here
  }
}