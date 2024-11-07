import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http'; // Import HttpClientModule
import { AppRoutingModule } from './app-routing.module'; // Import AppRoutingModule
import { AppComponent } from './app.component';
import { ImageSelectorComponent } from './components/image-selector/image-selector.component';
import { MainViewComponent } from './components/main-view/main-view.component';
import { AppIntroductionComponent } from './components/app-introduction/app-introduction.component';

@NgModule({
  declarations: [
    AppComponent,
    ImageSelectorComponent,
    MainViewComponent,
    AppIntroductionComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    HttpClientModule, // Add HttpClientModule
    AppRoutingModule, // Add AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}

